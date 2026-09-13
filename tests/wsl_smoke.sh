#!/usr/bin/env bash
# End-to-end smoke test of the trainer on a Linux/WSL ComfyUI install.
#   COMFY_ROOT=~/ComfyUI DATA=/path/to/songs bash tests/wsl_smoke.sh
set -u
COMFY_ROOT="${COMFY_ROOT:-$HOME/ComfyUI}"
DATA="${DATA:?set DATA to a folder with a couple of songs}"
OUT="${OUT:-$HOME/yue2_test_out}"
CACHE="${CACHE:-$HOME/yue2_cache}"
PY="${PY:-$COMFY_ROOT/venv/bin/python}"
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
KEEP='step |saved|done|Error|error|trainer|transcribed|Device'
ONLY="${ONLY:-all}"   # all | multi (skip schema/equivalence/single-GPU)

if [ "$ONLY" = "all" ]; then
echo "##### schemas"
"$PY" -W ignore tests/verify_node_schemas.py --comfy-root "$COMFY_ROOT" 2>&1 | grep -E 'json|ALL OK|FAILED|Error'
echo "##### forward equivalence"
"$PY" -W ignore tests/verify_against_comfy.py --comfy-root "$COMFY_ROOT" 2>&1 | grep -E '^\[|ALL OK|FAILED|Error'
fi
echo "##### acoustic --devices all (batch 2)"
time "$PY" -W ignore train_cli.py acoustic --comfy-root "$COMFY_ROOT" --data "$DATA" --cache-dir "$CACHE" --max-seconds 90 \
  --steps 6 --batch-size 2 --segment-seconds 20 --rank 8 --alpha 8 --devices all --out "$OUT/acoustic_all.safetensors" 2>&1 | grep -E "$KEEP"
if [ "$ONLY" = "all" ]; then
echo "##### acoustic --devices cuda:1 (batch 2)"
time "$PY" -W ignore train_cli.py acoustic --comfy-root "$COMFY_ROOT" --data "$DATA" --cache-dir "$CACHE" --max-seconds 90 \
  --steps 6 --batch-size 2 --segment-seconds 20 --rank 8 --alpha 8 --devices cuda:1 --out "$OUT/acoustic_gpu1.safetensors" 2>&1 | grep -E "$KEEP"
fi
echo "##### planner --devices all (transcribe, batch 2)"
time "$PY" -W ignore train_cli.py planner --comfy-root "$COMFY_ROOT" --data "$DATA" --cache-dir "$CACHE" --max-seconds 90 \
  --transcribe melody --steps 6 --batch-size 2 --max-tokens 1024 --rank 8 --alpha 8 --devices all --out "$OUT/planner_all.safetensors" 2>&1 | grep -E "$KEEP"
echo "##### verify files"
for f in acoustic_all acoustic_gpu1 planner_all; do
  echo "== $f"
  "$PY" -W ignore tests/verify_lora_file.py --comfy-root "$COMFY_ROOT" --lora "$OUT/$f.safetensors" 2>&1 | grep -E 'lora tensors|warnings|delta|ALL OK|FAILED|Error'
done
