"""S14 standalone no-network MCP stdio demo server.

For local acceptance only. Supports two harmless tools: echo, uppercase.
Does not read files, execute commands, access .env or use an API key.
"""
import json
import sys


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def reply(request, result):
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})


def tool_spec(name, description):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    }


for line in sys.stdin:
    try:
        request = json.loads(line)
        method = request.get("method")
        if method == "initialize":
            reply(request, {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "evome-s14-demo", "version": "1.0"},
            })
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            reply(request, {"tools": [
                tool_spec("echo", "Return the provided text unchanged"),
                tool_spec("uppercase", "Convert supplied text to uppercase"),
            ]})
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            value = arguments.get("text") if isinstance(arguments, dict) else None
            if name not in {"echo", "uppercase"} or not isinstance(value, str):
                reply(request, {
                    "isError": True,
                    "content": [{"type": "text", "text": "unknown tool or invalid text"}],
                })
                continue
            output = value if name == "echo" else value.upper()
            reply(request, {
                "isError": False,
                "content": [{"type": "text", "text": output[:4000]}],
            })
        elif "id" in request:
            send({
                "jsonrpc": "2.0", "id": request["id"],
                "error": {"code": -32601, "message": "Method not found"},
            })
    except (ValueError, TypeError, KeyError):
        # Protocol errors are server-local; never emit debug output on stdout.
        continue
