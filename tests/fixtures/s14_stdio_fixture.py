"""Offline MCP stdio fixture for the S14 tests. Never reads project secrets."""
import json
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

def send(msg):
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def response(msg, result=None, error=None):
    frame = {"jsonrpc": "2.0", "id": msg["id"]}
    frame["error" if error is not None else "result"] = error if error is not None else result
    send(frame)


def tool(name="echo"):
    return {
        "name": name, "description": "Fixture echo tool, no side effects",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    }


for line in sys.stdin:
    try:
        msg = json.loads(line)
        method = msg.get("method")
        if method == "initialize":
            response(msg, {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "s14-test-fixture", "version": "1"},
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            if MODE == "hang":
                time.sleep(3)
            elif MODE == "bad_metadata":
                response(msg, {"tools": [{"name": "../escape"}]})
            elif MODE == "duplicate":
                response(msg, {"tools": [tool(), tool()]})
            elif MODE == "pagination":
                if not msg.get("params", {}).get("cursor"):
                    response(msg, {"tools": [tool()], "nextCursor": "page2"})
                else:
                    response(msg, {"tools": [tool("uppercase")]})
            elif MODE == "collision":
                response(msg, {"tools": [tool()]})
            elif MODE == "bogus_jsonrpc":
                sys.stdout.write("NOT_JSON\n")
                sys.stdout.flush()
            else:
                response(msg, {"tools": [tool()]})
        elif method == "tools/call":
            if MODE == "reverse":
                send({
                    "jsonrpc": "2.0", "id": "secret-request",
                    "method": "sampling/createMessage",
                    "params": {"messages": []},
                })
                deny = json.loads(sys.stdin.readline())
                assert deny.get("error", {}).get("code") == -32601
            if MODE == "crash":
                sys.exit(0)
            if MODE == "tool_error":
                response(msg, {
                    "content": [{"type": "text", "text": "error"}],
                    "isError": True,
                })
                continue
            arguments = msg.get("params", {}).get("arguments", {})
            if msg.get("params", {}).get("name") == "uppercase":
                output = str(arguments.get("text", "")).upper()
            else:
                output = str(arguments.get("text", ""))
            response(msg, {
                "content": [{"type": "text", "text": output}],
                "isError": False,
            })
        elif "id" in msg:
            response(msg, error={"code": -32601, "message": "method not found"})
    except Exception as exc:
        if "id" in locals().get("msg", {}):
            response(msg, error={"code": -32000, "message": type(exc).__name__})
