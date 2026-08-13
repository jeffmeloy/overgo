# Overgo tightening

Goal: recipes and compiled neutral programs are the sole runtime authorities.
Each wave migrates every caller, deletes the displaced family path, preserves
numerics, passes relevant host/CUDA gates, commits through `cmd/gate`, then
merges `master`.

| Order | Scope | Deletion condition | Status |
|---:|---|---|---|
| 1 | Single-head MTP | Qwen/Cohere runtime APIs and codecs absent | Done |
| 2 | Multi-head MTP | Step/HY-V3 wrappers absent; one compiled coordinator | Done |
| 3 | Draft projections | family draft input/output builders absent | Done |
| 4 | Session forwards | DFlash/Eagle/Gemma assistant wrappers absent | Done |
| 5 | Encoder/decoder | T5 family block builders absent | Done |
| 6 | Layer execution | family leaf policies/builders absent; encoder, recurrent, latent, cache, and side-input selectors neutralized | Active |
| 7 | Weight catalogs | family loaders replaced by indexed catalog plans | Open |
| 8 | Runtime admission | architecture probes/fallbacks absent | Open |

Live ordering and executable verification remain in `docs/plan.json`.
