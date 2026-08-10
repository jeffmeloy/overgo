# overgo_gui — plan

A web GUI for **overgo**, built in the spirit of `adaptive_new`'s `adaptive_gpt.html`
approach: a **thin client over the Go server**, all state living server-side, no build
step, no external dependencies, one self-contained asset tree served by overgo itself.

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
  webui/                           # the embedded client (no build step)
    index.html                     # landing/probe card (port of adaptive_gpt.html)
    app.html                       # console shell (tabs + key field + model card)
    style.css                      # shared palette + components (from adaptive styles)
    boot.js                        # probe, router, fetch helpers, auth injection
    mod/
      chat.js                      # streaming chat console        (M2)
      status.js                    # model/props/slots/metrics     (M3)
      tokens.js                    # tokenize/detokenize/template   (M3)
      complete.js                  # completion + infill playground (Phase 2)
      embed.js                     # embeddings + rerank            (Phase 2)
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

---

## 6. Milestones & commit plan (commits land in this worktree)

- **M0 — plan.** This document. *(commit: "overgo_gui: plan")*
- **M1 — serve plumbing + landing.** `webui_static.go` (+test), `index.html`, `style.css`,
  `boot.js` probe. Verify `go run ./cmd/server <model>` serves `/` and the card enters on
  `/health`. *(commit)*
- **M2 — chat console.** `app.html` shell + `mod/chat.js` streaming. *(commit)*
- **M3 — status + tokenizer.** `mod/status.js`, `mod/tokens.js`. *(commit)*
- **Phase 2 milestones** — completion/infill, embeddings/rerank, responses. *(commits each)*

Each milestone is a green commit: `go build ./...` + `go test ./internal/server/...` pass,
and a manual smoke against a real model.

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

---

## 9. Decisions requested from the user

1. **D1:** OK to serve the GUI by embedding it into overgo's server (option A, touches
   `internal/server` additively)? Or keep the server untouched and ship a standalone page +
   CORS (option B)?
2. **Visual fidelity:** clone adaptive's exact palette/wordmark (sibling look), or a fresh
   overgo identity reusing only the structure?
3. **Scope confirmation:** MVP = chat + status + tokenizer (+ Phase 2 embeddings/rerank/infill),
   media explicitly deferred until the server exposes it — agreed?
