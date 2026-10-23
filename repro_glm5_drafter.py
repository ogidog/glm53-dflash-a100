# Standalone CPU repro / verification: GLM-5.3-Flash hybrid specs + a
# block-diffusion (DFlash2) drafter through vLLM's KV cache grouping.
#
#   CUDA_VISIBLE_DEVICES="" PYTHONPATH=/path/to/vllm python repro_glm5_drafter.py
#
# What it checks, per tree generation:
#
#  * Fork base (wtdcode/vllm-backport v0.13.0): the model-specific builder
#    `_get_kv_cache_groups_glm5_next` must accept the drafter's sliding-window
#    layers. Unpatched it declines them (returns None) and the generic
#    fallback degrades badly (every group flagged as draft, prefix-cache reuse
#    disabled, padding waste) -> "BUG REPRODUCED" (vllm#61443 family). With
#    the grouping fix applied the builder returns groups -> "FIX PRESENT".
#
#  * Upstream main (post-#61455): the generic planner groups the drafter's
#    layers (tagged `KVCacheSpec.is_draft`) into their own group and the
#    `is_draft` -> `is_eagle_group` tagging from vllm#61455 flags it
#    -> "FIX VERIFIED".
#
# Exit code: 0 when the tree behaves consistently with its generation,
# 1 on a contradiction, 2 when vllm cannot be probed.
import re
import sys
from types import SimpleNamespace

# Strip an editable-install finder if present so `import vllm` can only
# resolve to the checkout put first on PYTHONPATH.
sys.meta_path = [
    f for f in sys.meta_path
    if "vllm" not in getattr(type(f), "__module__", "")
    and "editable" not in getattr(type(f), "__name__", "").lower()
]

import torch  # noqa: E402

BF16 = torch.bfloat16
BS = 1280      # unified block (mamba-aligned), TP4
NUM_ATT = 11   # dense MLA + DSA indexer + kpool tail layers
NUM_KDA = 34   # KDA/mamba layers
KPOOL = 8      # index_kpool == indexer tokens_per_state
NUM_SPEC = 8   # DFlash2 block size n


def main():
    import vllm
    try:  # post-#61455 trees moved the planner into kv_cache_planning
        from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
    except ImportError:
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec, HiddenStateCacheSpec, MLAAttentionSpec,
        MambaSpec, SlidingWindowSpec,
    )
    try:
        from vllm.v1.kv_cache_layout import KVCacheLayout
        layouts = ("LBHNC", "BLHNC")
    except ImportError:
        KVCacheLayout = None
        layouts = ("FLAT",)

    print("vllm tree:", vllm.__file__)
    print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())

    def mk(cls, **kw):
        """Construct a spec, dropping per-field kwargs this tree lacks."""
        while True:
            try:
                return cls(**kw)
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                if m and m.group(1) in kw:
                    kw.pop(m.group(1))
                    continue
                raise

    # is_draft (vllm#61455) only exists on post-fix trees; probe it.
    try:
        SlidingWindowSpec(block_size=16, num_kv_heads=2, head_size=128,
                          head_size_v=128, dtype=BF16, sliding_window=2048,
                          is_draft=True)
        has_is_draft = True
    except TypeError:
        has_is_draft = False
    try:
        from vllm.v1.kv_cache_interface import KpoolTailSpec
    except ImportError:
        KpoolTailSpec = None
    print("planner generation:",
          "post-#61455 (is_draft)" if has_is_draft else "pre-#61455 (fork base)")

    def make_specs(with_draft, with_hidden):
        specs = {}
        for i in range(NUM_ATT):
            specs[f"layers.{4*i+3}.attn"] = mk(
                MLAAttentionSpec, block_size=BS, num_kv_heads=1,
                head_size=576, head_size_v=0, dtype=BF16)
            specs[f"layers.{4*i+3}.indexer"] = mk(
                MLAAttentionSpec, block_size=BS, num_kv_heads=1,
                head_size=128, head_size_v=0, dtype=torch.uint8,
                tokens_per_state=KPOOL)
            # Glm5NextTailCache: fork models the kpool tail as KpoolTailSpec
            # (a SlidingWindowSpec, which the fork builder aliases onto the
            # indexer page); main as a CircularBufferSpec ring.
            if KpoolTailSpec is not None:
                specs[f"layers.{4*i+3}.tail"] = mk(
                    KpoolTailSpec, block_size=BS, num_kv_heads=1,
                    head_size=128, head_size_v=0, dtype=torch.uint8,
                    sliding_window=KPOOL * 4, tokens_per_state=KPOOL)
            else:
                specs[f"layers.{4*i+3}.tail"] = mk(
                    CircularBufferSpec, block_size=KPOOL * 2, num_kv_heads=2,
                    head_size=128, head_size_v=0, dtype=BF16,
                    dcp_sharded=False)
        for i in range(NUM_KDA):
            specs[f"layers.{i}.kda"] = mk(
                MambaSpec, block_size=BS, shapes=((8, 128, 128), (8, 128, 256)),
                dtypes=(BF16, BF16), mamba_cache_mode="align")
        if with_draft:  # DFlash2: 5 GQA sliding-window layers, worker bs=16
            for i in range(5):
                specs[f"draft.layers.{i}.self_attn.attn"] = mk(
                    SlidingWindowSpec, block_size=16, num_kv_heads=2,
                    head_size=128, head_size_v=128, dtype=BF16,
                    sliding_window=2048,
                    **({"is_draft": True} if has_is_draft else {}),
                    dcp_sharded=False)
        if with_hidden:  # EAGLE3 aux hidden states: 5 streams x hidden 4096
            specs["layers.45.hidden"] = mk(
                HiddenStateCacheSpec, block_size=BS, num_kv_heads=5,
                head_size=4096, dtype=BF16)
        return specs

    def make_cfg(layout_name):
        layout = KVCacheLayout[layout_name] if KVCacheLayout else SimpleNamespace(
            is_block_outermost=False)
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=False),
            cache_config=SimpleNamespace(
                get_resolved_kv_cache_layout=lambda: layout, block_size=BS,
                num_gpu_blocks_override=None,
                prefix_cache_retention_interval=1280, mamba_cache_mode="align"),
            attention_config=SimpleNamespace(hisparse_config=None),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="glm5_next"),
                max_model_len=262144),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1, decode_context_parallel_size=1,
                enable_expert_parallel=True, tensor_parallel_size=4),
            speculative_config=SimpleNamespace(
                method="dflash", use_eagle=lambda: True,
                use_eagle_block_drop=lambda: True,
                num_speculative_tokens=NUM_SPEC),
            num_speculative_tokens=NUM_SPEC,
            device_config=SimpleNamespace(device_type="cpu"),
            max_in_flight_tokens=64,
            scheduler_block_size=BS,
        )

    if not has_is_draft:
        # ---- fork base: probe the model-specific builder directly ----
        try:
            from vllm.v1.core.kv_cache_utils import (
                _get_kv_cache_groups_glm5_next)
        except ImportError:
            print("PROBE INCOMPATIBLE: no _get_kv_cache_groups_glm5_next "
                  "and no is_draft; unrecognized tree")
            return 2
        base = _get_kv_cache_groups_glm5_next(
            make_cfg(layouts[0]), make_specs(False, False))
        print(f"builder without drafter: {'accepted' if base else 'DECLINED'}")
        with_draft = _get_kv_cache_groups_glm5_next(
            make_cfg(layouts[0]), make_specs(True, False))
        if with_draft is None:
            print("BUG REPRODUCED: the fork builder declines the DFlash2 "
                  "sliding-window layers; the generic fallback then flags "
                  "every group as draft (prefix-cache reuse disabled).")
            return 0
        print(f"FIX PRESENT: builder grouped the drafter into its own group "
              f"({len(with_draft)} groups, eagle-flagged: "
              f"{sum(bool(getattr(g, 'is_eagle_group', False)) for g in with_draft)}).")
        return 0

    # ---- post-#61455 main: full matrix through the public entry point ----
    failures = 0
    draft_ok = True
    for layout_name in layouts:
        for with_draft, with_hidden in ((False, False), (True, False),
                                        (True, True), (False, True)):
            tag = (f"layout={layout_name} draft={int(with_draft)}"
                   f" hidden={int(with_hidden)}")
            specs = make_specs(with_draft, with_hidden)
            try:
                groups = get_kv_cache_groups(make_cfg(layout_name), specs)
            except Exception as exc:  # noqa: BLE001 - the point of the probe
                failures += 1
                if with_draft:
                    draft_ok = False
                print(f"[RED]   {tag}: {type(exc).__name__}: {str(exc)[:220]}")
                continue
            note = ""
            if with_draft:
                draft_names = {n for n, s in specs.items()
                               if getattr(s, "is_draft", False)}
                dg = [g for g in groups
                      if not draft_names.isdisjoint(g.layer_names)]
                pure = all(all(n in draft_names for n in g.layer_names)
                           for g in dg)
                flagged = all(getattr(g, "is_eagle_group", False) for g in dg)
                note = (f" | draft groups: {len(dg)}, isolated: {pure}, "
                        f"eagle-flagged: {flagged}")
                if not dg or not pure:
                    draft_ok = False
                    failures += 1
            print(f"[GREEN] {tag}: {len(groups)} groups{note}")

    print("=" * 60)
    if draft_ok and failures == 0:
        print("FIX VERIFIED: post-#61455 planner groups the DFlash2 "
              "drafter correctly.")
        return 0
    print("UNEXPECTED: post-#61455 tree failed a case it should pass.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"PROBE INCOMPATIBLE: {e}")
        sys.exit(2)
