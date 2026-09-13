"""Import the node pack the way ComfyUI does and validate schemas + example workflows (no GPU needed)."""
import argparse, asyncio, importlib.util, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from train_cli import bootstrap_comfy
p = argparse.ArgumentParser(); p.add_argument("--comfy-root", required=True); a = p.parse_args()
bootstrap_comfy(a.comfy_root)
import comfy.sd  # noqa: F401  (ComfyUI imports its runtime before custom nodes)
import nodes as comfy_nodes
spec = importlib.util.spec_from_file_location("ComfyUI_YuE2_Trainer", HERE / "__init__.py", submodule_search_locations=[str(HERE)])
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
ext = asyncio.run(module.comfy_entrypoint())
node_list = asyncio.run(ext.get_node_list())
schemas = {}
for cls in node_list:
    schema = cls.define_schema()
    schema.validate() if hasattr(schema, "validate") else None
    ins = {i.id: i for i in schema.inputs}
    schemas[schema.node_id] = ins
    print(f"{schema.node_id}: {len(schema.inputs)} inputs, {len(schema.outputs)} outputs")
# core nodes referenced by the example workflows
import comfy_extras.nodes_yue2, comfy_extras.nodes_train, comfy_extras.nodes_audio  # noqa: F401
core_ids = set(comfy_nodes.NODE_CLASS_MAPPINGS)
for extra in ("comfy_extras.nodes_yue2", "comfy_extras.nodes_train", "comfy_extras.nodes_audio", "comfy_extras.nodes_audio_encoder", "comfy_extras.nodes_model_advanced", "comfy_extras.nodes_preview_any"):
    m = importlib.import_module(extra)
    if hasattr(m, "comfy_entrypoint"):
        e = asyncio.run(m.comfy_entrypoint())
        for cls in asyncio.run(e.get_node_list()):
            s = cls.define_schema(); core_ids.add(s.node_id); schemas[s.node_id] = {i.id: i for i in s.inputs}
    if hasattr(m, "NODE_CLASS_MAPPINGS"):
        core_ids.update(m.NODE_CLASS_MAPPINGS)
ok = True
for wf in sorted((HERE / "example_workflows").glob("*.json")):
    graph = json.loads(wf.read_text(encoding="utf-8"))
    problems = []
    for nid, node in graph.items():
        ct = node["class_type"]
        if ct not in core_ids and ct not in schemas:
            problems.append(f"{nid}: unknown node {ct}"); continue
        if ct in schemas:
            known = schemas[ct]
            for key, value in node["inputs"].items():
                if key not in known:
                    problems.append(f"{nid} ({ct}): unknown input {key}")
            for key, inp in known.items():
                if key not in node["inputs"] and not getattr(inp, "optional", False):
                    problems.append(f"{nid} ({ct}): missing input {key}")
        for key, value in node["inputs"].items():
            if isinstance(value, list) and value[0] not in graph:
                problems.append(f"{nid}: link to missing node {value[0]}")
    print(f"{wf.name}: {'OK' if not problems else problems}")
    ok &= not problems
print("ALL OK" if ok else "FAILED")
sys.stdout.flush()
import os; os._exit(0 if ok else 1)
