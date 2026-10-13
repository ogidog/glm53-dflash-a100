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

## Who needs this

| Your stack | Do you need it? |
|---|---|
| wtdcode/vllm-backport (~v0.13.0 base) + DFlash2/EAGLE drafter on A100 | **yes** — without it, startup fails |
| any vLLM base whose `kv_cache_utils.py` still has `_get_kv_cache_groups_glm5_next` (i.e. before upstream #57169, merged 2026-10-10) | yes |
| upstream main from 2026-10-10+ (Hopper/Blackwell) | no — #57169 landed a generic packed layout instead. Note you may still see the un-annotated draft group warning, vllm-project/vllm#61443 |
| plain upstream main on A100 | out of reach for now — no sm80 DSA backends upstream (see vllm-project/vllm#50576); that's why the fork exists |

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
production tree that has run for weeks).

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
- vllm-project/vllm#61443 — follow-up: on post-#57169 bases the dflash draft group is not
  annotated `is_eagle_group` (warning at startup; block-drop disabled).
- vllm-project/vllm#50576 — the SM8x support hub.

Licensed under Apache-2.0, same as vLLM.
