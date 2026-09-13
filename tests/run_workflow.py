"""Submit an API-format workflow to a running ComfyUI and wait for it (standard library only).

    python tests/run_workflow.py example_workflows/yue2_prepare_and_train_acoustic_api.json \
        --server http://127.0.0.1:8188 --set 2.folder=/path/to/songs --set 4.steps=4
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def call(server, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(server + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workflow")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--set", action="append", default=[], help="node.input=value (value parsed as JSON when possible)")
    p.add_argument("--max-wait", type=float, default=3600)
    args = p.parse_args()
    graph = json.loads(Path(args.workflow).read_text(encoding="utf-8"))
    for item in args.set:
        key, value = item.split("=", 1)
        node, inp = key.split(".", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        graph[node]["inputs"][inp] = value
    client = uuid.uuid4().hex
    out = call(args.server, "/prompt", {"prompt": graph, "client_id": client})
    if "prompt_id" not in out:
        print("validation error:", json.dumps(out, indent=1)[:3000])
        return 1
    pid = out["prompt_id"]
    print("queued", pid, flush=True)
    start = time.time()
    while time.time() - start < args.max_wait:
        time.sleep(5)
        hist = call(args.server, f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            print("status:", status.get("status_str"), f"({time.time() - start:.0f}s)")
            for node_id, outputs in entry.get("outputs", {}).items():
                for key, value in outputs.items():
                    if key in ("text", "string") or isinstance(value, list) and value and isinstance(value[0], str):
                        print(f"[node {node_id}] {key}:", "\n".join(str(v) for v in value)[:2500])
            if status.get("status_str") != "success":
                for msg in status.get("messages", []):
                    if msg[0] == "execution_error":
                        print("ERROR:", msg[1].get("node_type"), msg[1].get("exception_message"))
                        print("\n".join(msg[1].get("traceback", [])[-8:]))
                return 1
            return 0
    print("timed out")
    return 1


if __name__ == "__main__":
    sys.exit(main())
