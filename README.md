# GLM-5.3-Flash + DFlash2 on A100 (sm80) — the missing KV-cache grouping fix

**TL;DR:** [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) gives you the
sm80 DSA backends that GLM-5.3-Flash needs on A100. But on that fork (and any base before
upstream vllm-project/vllm#57169), starting the server with a **DFlash2 / EAGLE-family
drafter** dies during KV-cache planning:

```
NotImplementedError: indexer.k_cache: page size is not divisible by the maximum
page size and cannot be padded. Padding is only supported for non-MLA attention layers
```

This repo provides the fix: a pure-Python, single-file patch that teaches the GLM-5.3 KV
cache builder to give the drafter's sliding-window layers their own group. No rebuild,
no kernel changes. Running in production on 4x A100-80GB for weeks under real agent traffic.

## Status (2026-10-20)

- **Upstream:** the GLM-5.3-specific builder this patch targets was removed upstream
  (#57169 + #61183), and `is_eagle_group` is now derived from explicit draft-layer tags
  (#61455, merged 2026-10-17; regression test #62180). Our upstream PR #59656 was closed
  as superseded, and the follow-up issue #61443 was closed by #61455. The original
  `NotImplementedError` no longer reproduces on current upstream `main`.
- **So this patch is needed only on pre-#57169 bases** — in practice the
  `wtdcode/vllm-backport` fork and its `v0.13.1` / `latest-sm80` images.
  Fork PR: https://github.com/wtdcode/vllm-backport/pull/120 (open, awaiting a target
  branch/tag).
- **Patches re-verified `git apply`-clean on 2026-10-20** against the fork's `master`
  (`29e66dad`) and against the `v0.13.1` tag: 01 clean on both, 02 clean, 03 clean with
  one hunk offset in the test file.
- **Ampere note:** upstream still supports sm80 as a build target and ships sm80
  wheels/images. What is missing upstream is the sm80 DSA / sparse-MLA attention path for
  GLM-5.3-Flash / DeepSeek-V4-Flash (vllm-project/vllm#50576) — not the quantization
  format: NVFP4 weights do run on Ampere via the weight-only W4A16 Marlin fallback, while
  native FP4 compute, FP4 KV cache and FP4 attention stay Blackwell-only.

## Who needs this

| Your stack | Do you need it? |
|---|---|
| wtdcode/vllm-backport (~v0.13.0 base) + DFlash2/EAGLE drafter on A100 | **yes** — without it, startup fails |
| any vLLM base whose `kv_cache_utils.py` still has `_get_kv_cache_groups_glm5_next` (i.e. before upstream #57169, merged 2026-10-10) | yes |
| upstream main from 2026-10-10+ (Hopper/Blackwell) | no — #57169 landed a generic packed layout instead, and the un-annotated draft group warning is gone since #61455 (merged 2026-10-17) |
| plain upstream main on A100 | out of reach for now — no sm80 DSA backends upstream (see vllm-project/vllm#50576); that's why the fork exists. Note: upstream *does* support NVFP4/Ampere for *weights* (weight-only W4A16 via Marlin fallback, `docs/features/quantization/modelopt.md`), but native FP4 compute, FP4 KV cache and FP4 attention stay Blackwell-only — and the missing piece for GLM-5.3-Flash is the sm80 DSA/sparse-MLA attention backend, not the quantization format |

## The bug

GLM-5.3-Flash (`glm5_next`) packs its MLA + DSA-indexer layers via
`_get_kv_cache_groups_glm5_next`. Its gate requires every non-Mamba / non-`KpoolTailSpec`
layer to be exactly `MLAAttentionSpec`. A DFlash2 drafter contributes 5 GQA
`SlidingWindowSpec` layers (draft worker's own block size), the builder declines, and the
generic page-size unification then tries to pad every page up to the MLA page — which
MLA-family specs (the indexer) refuse. Startup dies.

## The fix (patches/)

- **`01_kv_cache_utils_glm5_drafter_grouping.patch`** — `vllm/v1/core/kv_cache_utils.py`:
  split `HiddenStateCacheSpec` (aux-eagle) and uniform drafter `SlidingWindowSpec` layers
  out of `attn_specs` (gated on `speculative_config.use_eagle()`); emit them as their own
  `KVCacheGroupSpec` re-aligned to the target block size with `is_eagle_group` (block drop)
  and `enable_kv_transfer=False`; append their regions after the packed MLA+indexer regions
  of the GLM block; charge them in bytes-per-block / max-memory accounting.
  Behavior is unchanged for models without a drafter.
- **`02_glm5next_model_eagle3.patch`** — `vllm/models/glm5next/nvidia/model.py`:
  `EagleModelMixin` + `SupportsEagle3` + aux hidden-state collection, required for the
  DFlash2 drafter to receive the target hidden states it was trained on.
- **`03_unit_tests.patch`** — two CPU unit tests for `tests/v1/core/test_kv_cache_utils.py`
  (RED on unpatched tree: no draft group / gate accepts non-EAGLE configs).

All three are `git apply`-clean on the fork's `master` (verified byte-for-byte against a
production tree that has run for weeks). Re-verified 2026-10-20 against the fork's current
`master` (`29e66dad`) and against the `v0.13.1` tag: all three still apply (01 clean on
both, 02 clean, 03 clean with a hunk offset in the test file).

## Install

Option A — apply to the tree (what the PR does):

```bash
cd ~/vllm-backport
git apply patches/01_kv_cache_utils_glm5_drafter_grouping.patch
git apply patches/02_glm5next_model_eagle3.patch
python -m py_compile vllm/v1/core/kv_cache_utils.py vllm/models/glm5next/nvidia/model.py
```

Option B — no-tree overlay (what we run in prod): copy the two patched files into a
`vllm/...` overlay directory and export `PYTHONPATH=/path/to/overlay`.

## Verify (CPU-only, no GPU/weights needed)

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/path/to/vllm python repro_glm5_drafter.py
# unpatched: RED — builder declines, generic path raises the indexer padding error
# patched:   GREEN — 1 draft group, is_eagle_group, no region overlaps
```

Unit tests (need the fork's venv):
`pytest tests/v1/core/test_kv_cache_utils.py -k drafter`

## Serving templates

`templates/serve_dflash.sh` and `templates/glm-5.3-flash.vllm-server.service` — the exact
flags verified in prod (TP4+EP4, ctx 262144, `--prefix-cache-retention-interval 1280`,
cudagraph capture sizes, `num_speculative_tokens=8`). Paths/user are placeholders.

One prod gotcha: the GLM block is mamba-aligned (1280 tokens at TP4 here), so
`prefix_cache_retention_interval` must stay a multiple of the unified block size.

## Production numbers (4x A100-80GB, TP4+EP4)

- GLM-5.3-Flash-AWQ-W4A16, ctx 262144, KV 1.38M tokens, DFlash2 `num_speculative_tokens=8`
- decode ~90–180 tok/s single-stream (~109 avg on agent traffic)
- spec acceptance 17–22% agent traffic / 26–58% long essays; stable for weeks
- prefix caching, reasoning & tool parsers, and KV connector unaffected

## Upstream trail

- vllm-project/vllm#59656 — this fix as an upstream PR (closed as superseded when #57169
  landed a generic version for post-Oct-10 bases).
- vllm-project/vllm#57169 / #61183 — upstream generic packed KV layout (why this patch is
  only needed on pre-#57169 bases like the fork).
- vllm-project/vllm#61443 — follow-up: on post-#57169 bases the dflash draft group was not
  annotated `is_eagle_group` (warning at startup; block-drop disabled). **Fixed upstream by
  #61455** (explicit draft-layer tags, merged 2026-10-17); regression test for this exact
  GLM-5.3-Flash + DFlash2 shape is in #62180.
- vllm-project/vllm#61455 / #62180 — upstream replacement of the `is_eagle_group` heuristics
  with explicit draft-layer tags, plus the regression test.
- vllm-project/vllm#50576 — the SM8x support hub. Upstream still ships sm80 wheels and
  images (`CUDA_ARCH_X86` includes 8.0/8.6) and still supports Ampere as a target, but has
  no sm80 DSA/sparse-MLA backends for GLM-5.3-Flash / DeepSeek-V4-Flash — that is why the
  fork (and this patch) still exist.

Licensed under Apache-2.0, same as vLLM.
