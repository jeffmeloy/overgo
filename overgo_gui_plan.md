# overgo_gui — plan

A web GUI for **overgo**, built in the spirit of `adaptive_new`'s `adaptive_gpt.html`
approach: a **thin client over the Go server**, all state living server-side, no build
step, no external dependencies, one self-contained asset tree served by overgo itself.

Beyond a chat/inference console, the GUI incorporates a **model-analysis workbench** with
the same capabilities as `C:\Users\jeffm\lophius` (a Python/Panel LLM research workbench) —
model & tokenizer inspection, a logit/probability/entropy lens, attention visualization, and
hidden-state projection — but reimplemented in **overgo's own stack**: analysis computed in
Go on the server, rendered with our vanilla-JS thin client (canvas/SVG, no plotly), **no
Python, torch, or transformers anywhere**. See §5A.

Worktree: `C:\Users\jeffm\overgo_gui` (branch `overgo_gui`). All work and commits happen
here, isolated from the `master` merge activity.

---

## 1. Doctrine (inherited from adaptive_gpt.html)

The reference (`C:\Users\jeffm\adaptive_new\adaptive_gpt.html` + `html/`, `html/shared/`) established:

- **Thin client, server owns state.** The page holds no model, no weights, no history it
  can't re-fetch. "all state lives in the server" is printed in the footer and meant literally.
- **Zero build.** Hand-written HTML + CSS custom properties + vanilla ES modules. No npm, no
  bundler, no framework, no CDN. It opens and runs.
- **Theme-aware.** Light/dark via `prefers-color-scheme` + a CSS variable palette
  (`--bg0/--ink/--acc/...`). We reuse the same palette so overgo reads as a sibling tool.
- **Probe-then-enter landing.** A launcher card polls the server; the moment it answers, it
  routes into the console. Server-down state shows the exact command to start it.
- **Served same-origin by the Go server.** In adaptive the server serves `html/`; the client
  fetches relative paths (`api/status`, `html/index.html`). Same-origin ⇒ no CORS dance.

We keep all five. The only things that change are the **endpoints** (overgo's API is
different) and the **feature surface** (overgo's server exposes a different capability set).

---

## 2. Ground truth: what overgo's server actually exposes today

Source: `cmd/server/main.go`, `internal/server/server.go` (route switch), `internal/server/server_admin.go`.

- **Bind:** `127.0.0.1:8080` by default (`-listen`). Single `http.Handler`, a flat
  `switch request.URL.Path`. **Unknown paths → 404 JSON.**
- **No static file serving.** There is no `/`, no `/html/`, no `embed.FS`, no file handler.
- **No CORS headers.** Confirmed: no `Access-Control-*` anywhere in `internal/server`.
  ⇒ a `file://` page fetching `http://localhost:8080` would be **blocked**. The GUI must be
  served same-origin (see §3, decision D1).

### Endpoint inventory

| Path | Handler | Auth¹ | GUI use |
|---|---|---|---|
| `/health`, `/healthz`, `/v1/health` | health → `{status, model}` | public | **landing probe** |
| `/models`, `/v1/models` | models → rich model meta (name, capabilities, `n_ctx`, `n_vocab`, `n_params`, ftype…) | public | model card |
| `/metrics` | Prometheus text | public | dashboard (optional) |
| `/props` | properties → server/model props | protected | dashboard |
| `/slots` | slotStatus → live slot/KV state | protected | dashboard |
| `/v1/chat/completions`, `/chat/completions` | chatCompletions (OpenAI, SSE stream) | protected | **chat console** |
| `/v1/completions`, `/completion`, `/completions` | completions / nativeCompletions | protected | completion playground |
| `/infill` | infill (FIM) | protected | infill playground |
| `/responses`, `/v1/responses` | responses (OpenAI Responses API) | protected | responses surface |
| `/v1/messages` (+ `/v1/messages/count_tokens`) | anthropicMessages | protected | (alt chat backend) |
| `/v1/embeddings`, `/embedding`, `/embeddings` | embeddings | protected | embeddings tool |
| `/rerank`, `/reranking`, `/v1/rerank` | rerank (if model supports) | protected | rerank tool |
| `/tokenize`, `/detokenize`, `/apply-template` | tokenizer utilities | protected | **tokenizer inspector** |
| `/*/input_tokens` | token counting | protected | live token counter |
| `/lora-adapters` | loraAdapters | protected | adapters panel |

¹ **Auth model:** if an API key is configured (`OVERGO_API_KEY` / `-api-key-file`), then `/v1/*`
(except `/v1/models`, `/v1/health`) and the listed native paths require `Authorization: Bearer …`.
`/health`, `/models`, `/metrics` are always public — so the landing page can probe with no key.

### What overgo's server does NOT expose (scope boundary)

overgo has rich internal media stacks (`internal/diffusionimage`, `internal/latentvideo`,
`internal/speechsynth`, `internal/oscillatorimage`) and CLI entrypoints (`cmd/diffusion`,
`cmd/latentvideo-run`, `cmd/generate`), but **none are wired into the HTTP server**. adaptive's
`media.html` image/video/TTS studio has **no overgo server counterpart yet**. Media is therefore
**out of scope for the GUI MVP** and gated behind future server endpoints (see §5, Phase 3).

---

## 3. Architecture decisions

### D1 — How the GUI is served ➜ **embed into overgo's server (recommended)**

Two options:

- **(A, recommended) Serve same-origin from the Go server via `embed.FS`.** Add a small
  static file handler to `internal/server` (a new `//go:embed webui` tree + a `/` and
  `/app/*` fall-through in the route switch, ahead of the 404 default). The client fetches
  same-origin relative paths; **no CORS, no separate process, one binary**. Exactly the
  adaptive model. Cost: a modest additive change to `internal/server` (isolated in the
  worktree; merges cleanly as new files + a couple of switch arms).
- **(B) Standalone `overgo_gui.html` opened from disk, + add CORS to the server.** Weaker:
  requires loosening the server's origin policy, splits "how do I run the UI" from "how do I
  run the server," and diverges from the adaptive doctrine. Rejected unless the user wants a
  fully static, server-untouched deliverable.

**Recommendation: A.** It is the faithful port and the better product. Flag for the user:
this means the `overgo_gui` branch touches `internal/server` (additively). If that's
undesirable before the master merge settles, fall back to B.

### D2 — Landing/probe ➜ `/health`, then `/models`

Mirror `adaptive_gpt.html`'s launcher card, repointed:
- Probe `GET /health` (public, cheap) on a 4 s interval with a 2.5 s timeout.
- On success, `GET /models` to fill the model card (name, ctx length, capabilities → which
  tabs light up: `rerank` cap gates the rerank tool, etc.).
- On failure, show the start command: `go run ./cmd/server -listen 127.0.0.1:8080 <model.gguf>`.

### D3 — Auth ➜ optional bearer field, stored client-side

A single "API key" input in the shell; persisted to `localStorage`; injected as
`Authorization: Bearer` on protected calls. Empty is fine when the server runs keyless
(the common local-lab case). A 401 surfaces a clear "key required / wrong key" banner.

### D4 — App shape ➜ lean single-page shell with tabs

adaptive uses multi-page (`index/infer/train/...`) + `html/shared/*.js` modules. overgo's MVP
surface is smaller, so we start with **one `app.html` shell** + a tiny router and per-tab ES
modules under `webui/mod/`, reusing adaptive's palette and component idioms (probe dot, cards,
mono code chips). This keeps parity of *feel* without porting adaptive's much larger surface.

### D5 — Streaming ➜ OpenAI-compatible SSE

`/v1/chat/completions` with `stream:true` emits `data:` chunks (`choices[].delta.content`),
terminated by `data: [DONE]`. The chat module consumes it with `fetch` + a `ReadableStream`
line parser (no EventSource, since we need POST + auth headers).

---

## 4. Directory layout (in this worktree)

```
overgo_gui_plan.md                 # this file
internal/server/
  webui_static.go                  # //go:embed webui/**, static handler, route wiring
  webui_static_test.go             # serves index, 404s unknown, no-auth on assets
  analyze_model.go                 # GET /analyze/model — arch/stats aggregate     (A1)
  analyze_vocab.go                 # GET /analyze/vocab — paged/searchable vocab    (A1)
  analyze_logits.go                # entropy field on native completion; capture    (A1)
  analyze_states.go                # opt-in hidden-state capture + Go PCA           (A2)
  analyze_attention.go             # opt-in analysis attention path + scores        (A3)
  analyze_*_test.go                # per-endpoint shape/auth tests
  webui/                           # the embedded client (no build step)
    index.html                     # landing/probe card (port of adaptive_gpt.html)
    app.html                       # console shell (tabs + key field + model card)
    style.css                      # shared palette + components (from adaptive styles)
    boot.js                        # probe, router, fetch helpers, auth injection
    viz.js                         # shared canvas/SVG primitives (heatmap, scatter, bars)
    mod/
      chat.js                      # streaming chat console            (M2)
      status.js                    # model/props/slots/metrics         (M3)
      tokens.js                    # tokenize/detokenize/template       (M3)
      complete.js                  # completion + infill playground     (Phase 2)
      embed.js                     # embeddings + rerank                (Phase 2)
      analyze_model.js             # stats + architecture summary       (A1)
      analyze_vocab.js             # searchable vocab + chat template   (A1)
      analyze_logits.js            # logit/prob/entropy lens            (A1)
      analyze_states.js            # hidden-state 2D/3D projection      (A2)
      analyze_attention.js         # attention heatmaps                 (A3)
```

Everything under `webui/` is embedded, so `go run ./cmd/server <model>` is the only step to
get the UI at `http://localhost:8080/`.

---

## 5. Feature surface — phased

### MVP (this effort)
1. **Landing + probe** (`index.html`) — the adaptive launcher, repointed to `/health`+`/models`.
2. **Chat console** (`mod/chat.js`) — streaming `/v1/chat/completions`; system prompt,
   temp/top-p/top-k/max-tokens controls, multi-turn history held client-side and replayed
   (server is stateless per request unless Responses is used), stop/regenerate, token/latency readout.
3. **Status dashboard** (`mod/status.js`) — `/models`, `/props`, `/slots`, `/metrics`:
   model identity, context length, live slot/KV occupancy, throughput.
4. **Tokenizer inspector** (`mod/tokens.js`) — `/tokenize`, `/detokenize`, `/apply-template`:
   paste text → token ids/pieces, round-trip, chat-template preview.

### Phase 2 (fast follow, same doctrine)
5. **Completion + infill playground** — `/completion`, `/infill` (FIM prefix/suffix/middle).
6. **Embeddings + rerank** — `/embedding`, `/rerank`: vectors, cosine similarity matrix,
   rank a candidate list against a query (gated on the `rerank` capability flag from `/models`).
7. **Responses surface** — `/responses` with server-side continuation (`store`/`previous_response_id`).

### Phase 3 (blocked on server work — not in this GUI effort)
8. **Media studio** (image/video/tts) — requires new overgo HTTP endpoints fronting
   `internal/diffusionimage` / `internal/latentvideo` / `internal/speechsynth`. Tracked here as
   a known gap; the GUI leaves a disabled "studio" entry that explains it's server-gated.

### Analysis track (parallel — see §5A for the full port)
9. **Analysis workbench** (the lophius capability port): model/vocab inspection + logit/
   probability/entropy lens (Tier 1, ships with MVP-class effort), then hidden-state projection
   (Tier 2) and attention heatmaps (Tier 3). Tier 1 is server-only; Tier 2/3 need engine taps.

---

## 5A. Analysis workbench — lophius capability port

Goal: the analytical power of lophius (`C:\Users\jeffm\lophius`), delivered through overgo's
server + thin client. Lophius runs on HF Transformers/PyTorch and visualizes with plotly; we
reproduce the *capabilities*, not the stack. Everything below is **Go on the server, vanilla
JS on the client** — entropy and PCA are a few dozen lines of Go, heatmaps and scatter plots
are canvas/SVG.

### Lophius capability inventory (from its docs + `outputs.py`/`models.py`)

- Model **statistics**, **architecture** tree, config; **tokenizer** info, **chat template**,
  searchable **vocabulary**; source links per module.
- Per-step **top-k logits**, **top-k probabilities**, full-vocabulary **entropy** (nats).
- **Attention scores** for any layer & head (heatmap).
- **First-token hidden states** in 2D/3D via PCA / t-SNE / UMAP / PaCMAP (scatter).
- Load **multiple models** at once and compare; edit config with **live reload**.

### Mapping to overgo — tiered by how much engine work each needs

**Tier 1 — data already in overgo, or a small localized server add (ships with the GUI):**

| Lophius capability | overgo source of truth | Server surface | Client module |
|---|---|---|---|
| Model statistics & architecture | `internal/model/architecture.go` (family/capability taxonomy), GGUF metadata, `/models`+`/props` | new `GET /analyze/model` (read-only aggregate) | `mod/analyze_model.js` — stat cards, architecture/composition summary |
| Tokenizer info + chat template | `internal/tokenizer` (`Vocab`), `/apply-template` | reuse `/apply-template`; new `GET /analyze/vocab?query=&page=` (paged/searchable) | `mod/analyze_vocab.js` — searchable vocab table, token flags, template viewer |
| **Logit / probability / entropy lens** | **`/completion` already returns `n_probs`** (top-N token logprobs per position); server already holds full `logits[]`+`logNormalization` (`server_native_generation.go`) | extend native completion with an `entropy` field per position (full-vocab, computed server-side); optionally an `analysis:true` flag that also captures the **prompt** positions | `mod/analyze_logits.js` — per-token top-k bars, logprob table, **entropy sparkline** across the sequence, "logit lens" strip |

Tier 1 is the marquee interpretability surface and is ~80% already present — the logit/prob
data flows today; only full-vocab **entropy** is a genuinely new (small) server value.

**Tier 2 — needs activation taps in the graph executor (fast-follow):**

| Lophius capability | Gap in overgo | Plan |
|---|---|---|
| Hidden-state capture (post-layer residual) | The executor streams activations through device memory without retaining per-layer copies | Add an opt-in "capture" mode that copies post-layer hidden states off-device for a single analysis request (bounded: first token / selected positions) |
| 2D/3D projection of hidden states | No projector | **PCA implemented in Go** (pure linear algebra, zero deps) → client renders a canvas/SVG scatter, color by token/position/layer. t-SNE/UMAP/PaCMAP are heavier iterative methods → **stretch** (a small Barnes-Hut t-SNE in Go later; PCA covers the MVP) |

Client: `mod/analyze_states.js` — layer selector, 2D/3D scatter, projection-method dropdown
(PCA enabled first).

**Tier 3 — needs an analysis-mode attention path (stretch, largest engine cost):**

| Lophius capability | Gap in overgo | Plan |
|---|---|---|
| Attention-score heatmaps per layer/head | overgo's **flash-attention kernels never materialize** the `[heads][q][k]` score matrix (that's the point of flash attention) | Add an **eager/analysis attention path** (or a capture kernel) that emits the score matrix for a bounded prompt when an analysis request asks for it; strictly opt-in and length-capped |

Client: `mod/analyze_attention.js` — layer/head pickers, `[q][k]` heatmap (canvas), row-normalized.

### Deliberate divergences from lophius (call out, don't paper over)

- **Single served model.** overgo's server loads one model at startup. Lophius's "load many
  models / compare / batch across models" doesn't map without multi-model serving — **out of
  scope**. Cross-**prompt** comparison within the one served model *is* supported.
- **No live config-edit/reload.** overgo binds a GGUF at startup; mutating config and hot-
  reloading isn't in the model runtime — **out of scope** (inspection is read-only).
- **No Python escape hatch.** Lophius's selling point is "drop to raw torch/transformers."
  overgo's equivalent is its Go API and CLIs (`cmd/model-info`, `cmd/inspect-gguf`); the GUI
  links to those rather than exposing a REPL.

### Why this fits the doctrine

Each analysis call is a plain request → server computes → JSON back → client draws. No client
state beyond what a re-fetch rebuilds. Entropy, PCA, and (later) t-SNE run in Go where the
weights and logits already live; the browser only ever renders. That is the same thin-client
contract as the chat console, extended to interpretability.

---

## 6. Milestones & commit plan (commits land in this worktree)

- **M0 — plan.** This document. *(commit: "overgo_gui: plan")*
- **M1 — serve plumbing + landing.** `webui_static.go` (+test), `index.html`, `style.css`,
  `boot.js` probe. Verify `go run ./cmd/server <model>` serves `/` and the card enters on
  `/health`. *(commit)*
- **M2 — chat console.** `app.html` shell + `mod/chat.js` streaming. *(commit)*
- **M3 — status + tokenizer.** `mod/status.js`, `mod/tokens.js`. *(commit)*
- **A1 — analysis Tier 1.** `analyze_model.go`/`analyze_vocab.go` + entropy on native
  completion; `mod/analyze_model.js`, `analyze_vocab.js`, `analyze_logits.js`. The lophius
  logit/entropy lens + model/vocab inspection, all from data overgo already has. *(commits)*
- **A2 — hidden-state projection.** executor capture hook + Go PCA (`analyze_states.go`);
  `mod/analyze_states.js` scatter. *(commit)*
- **A3 — attention heatmaps.** analysis attention path (`analyze_attention.go`);
  `mod/analyze_attention.js`. Largest engine cost — scheduled last, opt-in, length-capped. *(commit)*
- **Phase 2 milestones** — completion/infill, embeddings/rerank, responses. *(commits each)*

Each milestone is a green commit: `go build ./...` + `go test ./internal/server/...` pass,
and a manual smoke against a real model. A2/A3 touch the inference/executor packages (not just
`internal/server`), so they carry more merge risk and coordination — see §8.

---

## 7. Testing

- **Go:** `webui_static_test.go` — index served at `/`, assets served without auth,
  unknown `/app/*` falls back to the shell (SPA), truly-unknown non-asset path still 404s,
  content-types correct. Keep the static handler ordered so it never shadows API routes.
- **Manual smoke:** `go run ./cmd/server -listen 127.0.0.1:8080 <model.gguf>`, open
  `http://localhost:8080/`, confirm probe→enter, a streamed chat turn, and the dashboards.
- **Keyless and keyed:** smoke both with and without `OVERGO_API_KEY` to exercise D3.

---

## 8. Risks & coordination

- **Touching `internal/server` during the master merge.** The other agent is active on
  `master`/`internal/cuda` + `internal/server` may move. Mitigation: keep the GUI change
  **additive and isolated** — one new file `webui_static.go` + a minimal route arm + an embed
  dir. No edits to existing handlers. Rebase is trivial. (If even that is unwanted pre-merge,
  fall back to D1-option-B, server untouched.)
- **API drift.** overgo's routes/payloads could change. The client centralizes every endpoint
  string and shape in `boot.js` so a drift is a one-file fix.
- **Capability variance by model.** `rerank`/embedding/multimodal depend on the loaded model;
  the GUI reads `/models` capabilities and hides/disables tabs accordingly rather than erroring.
- **Analysis Tier 2/3 reach into the engine.** Hidden-state capture (A2) and the attention
  analysis path (A3) touch `internal/inference` / `internal/cuda/executor` — the same cone the
  merge agent is rewriting (the flash-attention work I reviewed earlier). High conflict risk.
  Mitigation: land Tier 1 (A1, server-only) first and independently; defer A2/A3 until the CUDA
  merge settles, and design them as opt-in, bounded, off-the-hot-path capture so they can't
  perturb serving numerics or the bit-identity fixtures.

---

## 9. Decisions requested from the user

1. **D1:** OK to serve the GUI by embedding it into overgo's server (option A, touches
   `internal/server` additively)? Or keep the server untouched and ship a standalone page +
   CORS (option B)?
2. **Visual fidelity:** clone adaptive's exact palette/wordmark (sibling look), or a fresh
   overgo identity reusing only the structure?
3. **Scope confirmation:** MVP = chat + status + tokenizer (+ Phase 2 embeddings/rerank/infill),
   media explicitly deferred until the server exposes it — agreed?
4. **Analysis depth (§5A):** how far do we take the lophius port?
   - **(a) Tier 1 only** — model/vocab inspection + logit/probability/entropy lens. Server-only,
     no engine changes, ~80% already in overgo. Lands cleanly alongside the merge. *(recommended
     first step.)*
   - **(b) Tier 1 + Tier 2** — add hidden-state capture + PCA projection (t-SNE/UMAP a later stretch).
   - **(c) Full port incl. Tier 3** — add attention heatmaps (needs an analysis attention path;
     highest engine cost, most conflict with the CUDA merge).
   My recommendation: **build (a) now**, schedule (b)/(c) after the `master` CUDA merge settles.
