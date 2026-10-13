# Standalone CPU repro: GLM-5.3-Flash hybrid specs + a block-diffusion
# drafter's sliding-window layers crash the KV cache grouping on vLLM main.
#
#   CUDA_VISIBLE_DEVICES="" PYTHONPATH=/path/to/vllm python repro_glm5_drafter.py
#
# Expected on unpatched main: builder declines (returns None), the generic
# unify_kv_cache_spec_page_size then raises
#   NotImplementedError: ... indexer ... page size is not divisible by the
#   maximum page size and cannot be padded. Padding is only supported for
#   non-MLA attention layers
import torch
from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_glm5_next
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowSpec,
    MambaSpec,
    KpoolTailSpec,
)

BF16 = torch.bfloat16
BS = 1280          # unified block after mamba alignment
NUM_ATT = 11       # dense MLA + indexer + kpool tail layers
NUM_KDA = 34       # KDA/mamba layers
TPS = 8            # indexer tokens_per_state

specs = {}
for i in range(NUM_ATT):
    specs[f"layers.{4*i+3}.attn"] = MLAAttentionSpec(
        block_size=BS, num_kv_heads=1, head_size=576, head_size_v=0,
        dtype=BF16, cache_dtype_str=None)
    specs[f"layers.{4*i+3}.indexer"] = MLAAttentionSpec(
        block_size=BS, num_kv_heads=1, head_size=128, head_size_v=0,
        dtype=torch.uint8, cache_dtype_str=None, tokens_per_state=TPS)
    specs[f"layers.{4*i+3}.tail"] = KpoolTailSpec(
        block_size=BS, num_kv_heads=1, head_size=128, head_size_v=0,
        dtype=torch.uint8, sliding_window=TPS * 4, tokens_per_state=TPS)
for i in range(NUM_KDA):
    specs[f"layers.{i}.kda"] = MambaSpec(
        block_size=BS, shapes=((8, 128, 128), (8, 128, 256)),
        dtypes=(BF16, BF16))
# DFlash2 drafter: 5 GQA sliding-window layers at the draft worker's block size
for i in range(5):
    specs[f"draft.layers.{i}.self_attn.attn"] = SlidingWindowSpec(
        block_size=16, num_kv_heads=2, head_size=128, head_size_v=128,
        dtype=BF16, sliding_window=2048)

spec_cfg = SimpleNamespace(
    method="dflash", use_eagle=lambda: True,
    use_eagle_block_drop=lambda: True, num_speculative_tokens=8,
    use_multi_module_mtp=lambda: False)
cfg = SimpleNamespace(
    cache_config=SimpleNamespace(
        num_gpu_blocks_override=None, mamba_cache_mode="align",
        prefix_match_unit=16, prefix_cache_retention_interval=1280,
        block_size=BS),
    parallel_config=SimpleNamespace(
        pipeline_parallel_size=1, decode_context_parallel_size=1,
        enable_expert_parallel=True, tensor_parallel_size=4),
    speculative_config=spec_cfg,
    model_config=SimpleNamespace(max_model_len=262144),
    max_in_flight_tokens=64,
    scheduler_config=SimpleNamespace(max_num_seqs=4))

groups = _get_kv_cache_groups_glm5_next(cfg, dict(specs))
if groups is None:
    print("RED: _get_kv_cache_groups_glm5_next declined; the generic fallback "
          "then raises NotImplementedError on the indexer layer (see issue text)")
else:
    print("GREEN:", len(groups), "groups, draft group flagged:",
          [g.layer_names for g in groups if g.is_eagle_group])
