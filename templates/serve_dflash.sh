#!/bin/bash
# GLM-5.3-Flash AWQ W4A16, TP4+EP4 + DFlash2 drafter (retention 1280).
# Assumes the grouping/eagle3 fix is applied (merged PR or overlay on PYTHONPATH).
# Adjust: venv path, model/draft paths, port.
source "$HOME/vllm-backport/venv/bin/activate"
# Only needed if you use the overlay install instead of patching the tree:
# export PYTHONPATH="$HOME/vllm-dflash"
export NCCL_ALGO=Ring NCCL_PROTO=Simple
MODEL="$MODEL_DIR/GLM-5.3-Flash-AWQ-W4A16"
DRAFT="$DRAFT_DIR/GLM-5.3-Flash-DFlash2"
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,9,16,18],"max_cudagraph_capture_size":18}'
exec vllm serve "$MODEL" \
  --served-model-name GLM-5.3-Flash \
  --port 8382 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 262144 --gpu-memory-utilization 0.94 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --prefix-match-unit 16 --prefix-cache-retention-interval 1280 --enable-prompt-tokens-details --disable-custom-all-reduce \
  --compilation-config "$COMPILATION_CONFIG" \
  --speculative-config "{\"model\": \"$DRAFT\", \"num_speculative_tokens\": 8}" \
  --host 0.0.0.0 \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
