"""S14 MCP stdio transport and namespace-aware tool pool integration.

The only process-spawning entry is the operator's explicit "connect".
Servers are selected from a trusted workspace-local config (no shell,
no environment overrides, no inbound sampling requests). All discovered
tools, including tools advertised as read-only, require S03 approval.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from app.tools.base import ToolDefinition, ToolImplementation
from app.tools.registry import ToolRegistry

_SERVER_ID = re.compile(r"[a-z][a-z0-9_-]{0,23}\Z")
_TOOL_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")
_MAX_RPC_BYTES = 65536
_MAX_CONFIG_BYTES = 16384
_MAX_RESULT_BYTES = 16384
_MAX_TOOLS_SERVER = 24
_MAX_SERVERS = 4
_CLIENT_PROTOCOL = "2025-06-18"


def _name(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"S14 invalid {label}")
    return value


def _safe_json(value: Any, *, limit: int = _MAX_RPC_BYTES) -> bytes:
    raw = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    if len(raw) > limit:
        raise ValueError("S14 JSON payload exceeds size limit")
    return raw


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: str
    args: tuple[str, ...]

    @classmethod
    def parse(cls, name: str, row: Any) -> "MCPServerSpec":
        _name(name, _SERVER_ID, "server name")
        if not isinstance(row, dict) or set(row) != {"command", "args"}:
            raise ValueError("MCP server spec must contain command and args only")
        command = row["command"]
        args = row["args"]
        if (not isinstance(command, str) or not 1 <= len(command) <= 260
                or "\x00" in command or "\r" in command or "\n" in command):
            raise ValueError("invalid MCP executable")
        if (not isinstance(args, list) or len(args) > 24
                or any(not isinstance(arg, str) or len(arg) > 500
                       or "\x00" in arg or "\r" in arg or "\n" in arg
                       for arg in args)):
            raise ValueError("MCP args must be an array of bounded strings")
        return cls(name, command, tuple(args))


class MCPConfig:
    """Read-only config; never parses/seeks .env or accepts inline credentials."""

    def __init__(self, workspace_root: str | Path, filename: str = ".mcp_servers.json"):
        self.root = Path(workspace_root).resolve()
        relative = Path(filename)
        if (relative.is_absolute() or not filename or relative == Path(".")
                or ".." in relative.parts or relative.suffix != ".json"):
            raise ValueError("MCP config must be a workspace-relative JSON file")
        self.path = self.root / relative
        if not self.path.resolve().is_relative_to(self.root):
            raise PermissionError("MCP config path escapes workspace")

    def load(self) -> dict[str, MCPServerSpec]:
        target = self.path
        if target.is_symlink() or not target.resolve().is_relative_to(self.root):
            raise PermissionError("MCP config symlink/path escape denied")
        if not target.exists():
            return {}
        if target.stat().st_size > _MAX_CONFIG_BYTES:
            raise ValueError("MCP config file too large")
        try:
            obj = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise ValueError("MCP server config is invalid JSON") from exc
        if not isinstance(obj, dict) or set(obj) != {"servers"}:
            raise ValueError("MCP config requires one servers object")
        raw = obj["servers"]
        if not isinstance(raw, dict) or len(raw) > _MAX_SERVERS:
            raise ValueError("MCP config exceeds server limit")
        return {
            _name(name, _SERVER_ID, "server key"): MCPServerSpec.parse(name, value)
            for name, value in raw.items()
        }


def _minimal_process_env() -> dict[str, str]:
    """Don't automatically hand an external tool the EvoMe API key or .env."""
    allow = {
        "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
        "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA",
        "APPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432",
        "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV", "PYTHONHOME",
    }
    result = {
        key: value for key, value in os.environ.items()
        if key.upper() in allow
    }
    # Python stdio demo (and other Python-based servers) should encode MCP
    # JSON frames as UTF-8 on Windows regardless of terminal codepage.
    result["PYTHONIOENCODING"] = "utf-8"
    result["PYTHONUTF8"] = "1"
    return result


class MCPStdioClient:
    """Bounded synchronous JSON-RPC request channel over one stdio process."""

    def __init__(
        self, spec: MCPServerSpec, *, workspace_root: str | Path,
        timeout: float = 6.0,
    ) -> None:
        self.spec = spec
        self.workspace = Path(workspace_root).resolve()
        self.timeout = max(0.25, min(20.0, float(timeout)))
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._responses: queue.Queue[dict[str, Any] | BaseException] = queue.Queue(maxsize=32)
        self._lock = threading.RLock()
        self._message_id = 0
        self._ready = False

    def _reader_loop(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                raw = proc.stdout.readline(_MAX_RPC_BYTES + 1)
                if not raw:
                    raise EOFError("MCP stdio server closed stdout")
                if len(raw) > _MAX_RPC_BYTES:
                    raise ValueError("MCP server emitted oversized line")
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError) as exc:
                    raise ValueError("MCP server sent malformed JSON-RPC") from exc
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    raise ValueError("MCP server sent invalid JSON-RPC frame")
                self._responses.put_nowait(message)
        except BaseException as exc:
            try:
                self._responses.put_nowait(exc)
            except queue.Full:
                pass

    def _send(self, message: dict[str, Any]) -> None:
        proc = self._process
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise ConnectionError("MCP server is not running")
        data = _safe_json(message)
        try:
            proc.stdin.write(data + b"\n")
            proc.stdin.flush()
        except (OSError, ValueError, BrokenPipeError) as exc:
            raise ConnectionError("MCP server stdin disconnected") from exc

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._message_id += 1
            expected = self._message_id
            self._send({
                "jsonrpc": "2.0", "id": expected,
                "method": method, "params": params,
            })
            import time
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"MCP request timed out: {method}")
                try:
                    incoming = self._responses.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError(f"MCP request timed out: {method}") from exc
                if isinstance(incoming, BaseException):
                    raise ConnectionError(f"MCP transport closed: {type(incoming).__name__}") from incoming
                if "id" not in incoming:
                    # MCP notifications are untrusted data, never host actions.
                    continue
                if "method" in incoming:
                    # Never grant reverse model sampling, elicitation, or roots.
                    self._send({
                        "jsonrpc": "2.0", "id": incoming["id"],
                        "error": {"code": -32601, "message": "Unsupported client request"},
                    })
                    continue
                if incoming.get("id") != expected:
                    raise ValueError("MCP response id mismatch")
                if "error" in incoming:
                    err = incoming["error"]
                    raise RuntimeError(
                        f"MCP server RPC error: {str(err)[:300]}"
                    )
                result = incoming.get("result")
                if not isinstance(result, dict):
                    raise ValueError("MCP response must have a result object")
                if len(_safe_json(result)) > _MAX_RPC_BYTES:
                    raise ValueError("MCP response too large")
                return result

    def connect(self) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError("MCP server already connected")
            executable = self.spec.command
            # The user explicitly chooses a trusted executable; no shell.
            resolved = shutil.which(executable)
            if resolved is None:
                raise FileNotFoundError("configured MCP executable was not found")
            self._process = subprocess.Popen(
                [resolved, *self.spec.args],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=self.workspace,
                env=_minimal_process_env(), shell=False, bufsize=0,
            )
            self._reader = threading.Thread(
                target=self._reader_loop, name=f"mcp-{self.spec.name}-stdout",
                daemon=True,
            )
            self._reader.start()
            try:
                result = self._request("initialize", {
                    "protocolVersion": _CLIENT_PROTOCOL,
                    "capabilities": {},
                    "clientInfo": {"name": "EvoMe", "version": "0.3.14"},
                })
                if (not isinstance(result.get("protocolVersion"), str)
                        or not isinstance(result.get("serverInfo"), dict)
                        or not isinstance(result.get("capabilities"), dict)):
                    raise ValueError("MCP initialize response missing required fields")
                self._send({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                })
                self._ready = True
            except Exception:
                self.close()
                raise

    def list_tools(self) -> list[dict[str, Any]]:
        if not self._ready:
            raise ConnectionError("MCP server not initialized")
        seen: set[str] = set()
        cursor: str | None = None
        output: list[dict[str, Any]] = []
        for _ in range(4):
            params = {"cursor": cursor} if cursor is not None else {}
            result = self._request("tools/list", params)
            tools = result.get("tools")
            if not isinstance(tools, list):
                raise ValueError("invalid MCP tools/list response")
            for row in tools:
                if not isinstance(row, dict):
                    raise ValueError("invalid MCP tool entry")
                name = _name(row.get("name"), _TOOL_ID, "tool name")
                if name in seen or len(output) >= _MAX_TOOLS_SERVER:
                    raise ValueError("duplicate or excessive MCP tool names")
                desc = row.get("description", "")
                schema = row.get("inputSchema", {"type": "object", "properties": {}})
                if (not isinstance(desc, str) or len(desc) > 1600
                        or not isinstance(schema, dict)
                        or schema.get("type") != "object"
                        or not isinstance(schema.get("properties", {}), dict)
                        or len(_safe_json(schema, limit=9000)) > 9000):
                    raise ValueError("MCP tool metadata invalid")
                seen.add(name)
                output.append({
                    "name": name,
                    "description": desc[:500],
                    "schema": schema,
                })
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                return output
            if (not isinstance(next_cursor, str) or not next_cursor
                    or len(next_cursor) > 120 or next_cursor == cursor):
                raise ValueError("invalid MCP pagination cursor")
            cursor = next_cursor
        raise ValueError("MCP tool pagination limit exceeded")

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._ready:
            raise ConnectionError("MCP server is not initialized")
        _name(tool_name, _TOOL_ID, "tool name")
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be JSON object")
        _safe_json(arguments, limit=12000)
        try:
            response = self._request("tools/call", {
                "name": tool_name, "arguments": arguments,
            })
            if response.get("isError"):
                raise RuntimeError("MCP tool reported error (see server logs)")
            content = response.get("content")
            if not isinstance(content, list):
                raise ValueError("MCP tools/call missing content array")
            safe_content: list[dict[str, Any]] = []
            for part in content[:12]:
                if not isinstance(part, dict) or part.get("type") != "text":
                    # Do not forward arbitrary image bytes, executable payloads
                    # or remote resource blobs into the Agent context in S14.
                    continue
                data = part.get("text")
                if not isinstance(data, str):
                    raise ValueError("invalid MCP text content")
                safe_content.append({"type": "text", "text": data[:4000]})
            result = {"content": safe_content, "isError": False}
            _safe_json(result, limit=_MAX_RESULT_BYTES)
            return result
        except (TimeoutError, ConnectionError, ValueError):
            self.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._ready = False
            proc, self._process = self._process, None
            if proc is None:
                return
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=0.6)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                    proc.wait(timeout=0.6)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            try:
                if proc.stdout is not None:
                    proc.stdout.close()
            except OSError:
                pass


class MCPPluginManager:
    """Operator-managed MCP connections attached to an existing ToolRegistry."""

    def __init__(
        self, *,
        workspace_root: str | Path,
        registry: ToolRegistry,
        config_file: str = ".mcp_servers.json",
        approval_revoke: Callable[[set[str]], None] | None = None,
        timeout: float = 6.0,
    ) -> None:
        self.registry = registry
        self.config = MCPConfig(workspace_root, config_file)
        self._clients: dict[str, MCPStdioClient] = {}
        self._names: dict[str, list[str]] = {}
        self._tokens: dict[str, object] = {}
        self._lock = threading.RLock()
        self._revoke = approval_revoke
        self._timeout = timeout

    def servers(self) -> list[dict[str, Any]]:
        specs = self.config.load()
        return [
            {"name": name, "connected": name in self._clients,
             "tool_count": len(self._names.get(name, []))}
            for name in specs
        ]

    def connected_tools(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"server": name, "tool": fullname,
                 "description": self.registry.get_tool(fullname).description}
                for name, names in self._names.items()
                for fullname in names
            ]

    def connect(self, server: str) -> list[str]:
        _name(server, _SERVER_ID, "server name")
        with self._lock:
            if server in self._clients:
                raise ValueError("MCP server already connected")
            specs = self.config.load()
            if server not in specs:
                raise KeyError("server not declared in MCP config")
            client = MCPStdioClient(
                specs[server], workspace_root=self.config.root,
                timeout=self._timeout,
            )
            registered_names: list[str] = []
            try:
                client.connect()
                discovered = client.list_tools()
                names = [f"mcp__{server}__{item['name']}" for item in discovered]
                if any(name in {t.name for t in self.registry.list_tools()} for name in names):
                    raise ValueError("MCP tool name conflicts with existing registry")
                if self._revoke is not None:
                    self._revoke(set(names))
                token = object()
                # Register only after the complete list is validated.
                for item, fullname in zip(discovered, names):
                    schema = item["schema"]
                    # ToolDefinition accepts a simple properties-shaped schema;
                    # the full JSON Schema stays validated at MCP transport.
                    description = (
                        f"[External MCP server '{server}', untrusted; "
                        "S03 approval required] " + item["description"]
                    )[:600]
                    required = schema.get("required", [])
                    if (not isinstance(required, list)
                            or any(not isinstance(r, str) or r not in schema.get("properties", {})
                                   for r in required)):
                        raise ValueError("MCP tool schema required fields invalid")
                    self.registry.register_tool(ToolDefinition(
                        name=fullname,
                        description=(
                            description + f" Required: {', '.join(required)}"
                            if required else description
                        )[:650],
                        input_schema=dict(schema.get("properties", {})),
                    ))
                    registered_names.append(fullname)
                    def bound_handler(srv: str, tool: str, current_token: object):
                        # Values are captured in a closure, never accepted as
                        # caller-provided keyword arguments.
                        def invoke(**kwargs: Any):
                            return self._call_bound(srv, tool, current_token, kwargs)
                        return invoke
                    self.registry.register_implementation(ToolImplementation(
                        name=f"s14_{fullname}_v1",
                        tool_name=fullname,
                        handler=bound_handler(server, item["name"], token),
                    ))
                self._clients[server] = client
                self._names[server] = names
                self._tokens[server] = token
                return names
            except Exception:
                # Roll back every newly registered name without perturbing
                # built-in S01-S13 tools.
                for fullname in registered_names:
                    self.registry.unregister_tool(fullname)
                client.close()
                raise

    def _call_bound(
        self, server: str, tool: str, token: object, args: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if self._tokens.get(server) is not token or server not in self._clients:
                raise PermissionError("stale MCP tool binding; reconnect and approve again")
            client = self._clients[server]
        return client.call(tool, args)

    def disconnect(self, server: str) -> None:
        _name(server, _SERVER_ID, "server name")
        with self._lock:
            if server not in self._clients:
                raise KeyError("MCP server is not connected")
            names = self._names.pop(server, [])
            self._tokens.pop(server, None)
            client = self._clients.pop(server)
            if self._revoke is not None:
                self._revoke(set(names))
            for fullname in names:
                self.registry.unregister_tool(fullname)
            client.close()

    def close(self) -> None:
        with self._lock:
            for server in list(self._clients):
                self.disconnect(server)
