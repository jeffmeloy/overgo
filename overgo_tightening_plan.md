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
| 6 | Layer execution | architecture-family predicates and duplicate registries absent; alternate-prediction admission is topology-owned; family leaf policies/builders absent; ordered relative encoder/decoder stages plus sealed graph and normalization facts, neutral recurrent, latent, compressed-hyper, WKV, cache, side-input, draft topology/storage, weight catalogs, dense/expert, and rotary IR/selectors | Active |
| 7 | Weight catalogs | family loaders replaced by indexed catalog plans; empty-profile compatibility read removed; standard SwiGLU schema shared by recurrent and auxiliary readers; remaining staged requirement duplication absent | Active |
| 8 | Runtime admission | continuous batching, retained-cache, capture, sequence-cache, projected-input, specialized-output, diffusion, recurrent, and packed-device admission compiled; runtime profile probes absent | Done |

Live ordering and executable verification remain in `docs/plan.json`.
