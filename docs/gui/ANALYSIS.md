# Current workbench GUI: analysis record

Written 2026-09-04 for the professional GUI campaign (lane
`professional_overgo_gui`). This is a design record of what the present
GUI does, taken from the source at the lane base 9dce4fca, and of what a first-time
user cannot do with it. It is the inventory the campaign's retention rows
are checked against. Reports of evidence are generated; this document is
not one.

## Shape

The GUI is a thin client embedded in `cmd/server` under a same-origin
content-security policy with no build step and no external assets. All
state lives in the server. The files under `internal/server/webui/`:

| File | Role | Lines |
| --- | --- | --- |
| `index.html` | Landing card: probes `/health`, then links to the workbench | 35 |
| `app.html` | Workbench shell: sidebar sections, topbar, operation shell, panel host; loads 30 scripts | 65 |
| `boot.js` | `window.overgo`: fetch with bearer injection, SSE reader, DOM helper, tab registry, model info memo, model pill and live swap, API key handling | ~600 |
| `style.css` | One token set for dark and light themes; every component class | ~700 |
| `viz.js`, `md.js`, `workflow.js`, `operations_shell.js`, `schema_form.js`, `probe.js` | Charts, markdown, workflow graphs, long-operation strip, schema-driven forms, landing probe | |
| `mod/*.js` | 24 self-registering tabs, 3042 lines total; largest are agent (380), runtime (273), chat (241), training (228), peers (210), evaluations (205) | |

Routing is hash-based inside `app.html`. Each module calls
`overgo.registerTab({id, mount})`; the shell fetches `/workspace/manifest`,
which declares five sections and 24 tabs, each tab bound to a capability
the server evaluates against the served recipe and the store. A tab whose
capability the served model lacks is disabled with a refusal string, so the
shell hides what this model cannot do.

Sections and tabs from `workspace_manifest.json`:

- Inference: Chat, Agent, Generate, Images, Speech, Video, Video Edit, Runtime, Activity
- Library: Library (local catalog, Hugging Face search, verified downloads)
- Datasets: Datasets
- Training: Train, Model Builder, Export, Runs
- Workbench: Inbox, Automations, Peers, Recipe, Compositions, Artifacts, Evaluations, Model, Vocabulary, Logits, Hidden states, Attention, Tensors

## What works today

- **Model in view and live switch.** The topbar pill shows the served model.
  Behind the swap proxy (`cmd/swap`, launched by `overgo_gui.bat` on
  localhost:8080) the pill lists the store's servable catalog from
  `/catalog/models` and switches the served child through
  `/health?swap=<name>`; the proxy owns the child lifecycle and the store's
  writer lock admits one serving process. Without the proxy the pill
  explains that a relaunch is needed.
- **Chat with attachments.** `mod/chat.js` streams `/v1/chat/completions`,
  counts input tokens through `/v1/chat/completions/input_tokens`, shows
  prefill and decode tok/s from the response timings, and attaches images,
  WAV audio and MP4 video as the protocol's own content parts (`image_url`,
  `input_audio`, `input_video`) with no side channel. Messages live in a
  browser array; the interaction record on the server is what survives.
- **Generation workspaces.** Separate tabs drive `/v1/images/generations`,
  `/v1/audio/speech`, `/v1/videos/generations` and `/v1/videos/edits`.
- **Analysis inspectors.** Model, vocabulary, logits, hidden states,
  attention and tensors over the loaded artifact through the `/analyze/*`
  routes; the attention view recomputes bounded plain-causal attention on the
  host and refuses score policies it cannot reproduce.
- **Operations shell.** A strip of long operations with progress, cancel,
  recovery decisions and the durable receipt, fed by `/operations/evidence`.
- **Operator surfaces.** Agent sessions with native tool calling, the
  approvals inbox, automations, peers, recipes, compositions, artifacts,
  evaluations, training sessions with evidence brackets, runs, datasets and
  the model builder.
- **Acceptance.** `internal/server/webui_browser_acceptance_test.go` drives a
  real headless Chromium through `internal/webuilane` (DevTools socket):
  key storage, tab activation, schema forms, responsive layout at narrow
  widths, operation recovery and cancellation. `cmd/webui-lane` runs it and
  reports UNAVAILABLE without a browser.

## What a first-time user cannot do

1. **Start a conversation without knowing the tab layout.** The landing card
   offers only "enter workbench"; the shell opens on a sidebar of 24 tabs in
   five sections. Chat is one tab among nine in the Inference section.
2. **Choose a model with evidence at the point of choice.** The catalog with
   evidence lines lives in the Library tab; the pill lists names. The
   README describes evidence beside each entry in the picker, but the pill
   itself shows the name and a serve button.
3. **Keep a conversation.** `chat.js` holds its messages in memory; reloading
   the page or switching models loses the visible thread even though the
   server's interaction record retains it. There is no conversation list.
4. **Drop or paste media.** Attachments come only through a file picker with
   a fixed accept list (png, jpeg, gif, wav, mp4). No drag-and-drop, no paste,
   no documents (pdf, text, csv), no preview for audio or video, and refusal
   happens as a server error after sending rather than at attach time from
   the recipe's declared capabilities.
5. **Reach a modality from the conversation.** Image, video and speech
   generation are separate tabs with their own prompts; the conversation
   cannot ask for an image and receive it inline, and nothing renders a
   generated artifact in the thread.
6. **Investigate the turn just produced.** The inspectors take their own
   prompt inputs; there is no "inspect this response" that opens logits,
   hidden states or attention for the exact prompt and completion on screen.
7. **See a running switch or job from the conversation.** The operations
   shell is part of the workbench shell; the landing page and a chat-only
   view would not show a swap in progress.
8. **Use it from the keyboard or on a phone with confidence.** The
   acceptance lane checks one narrow-width layout rule and focus styling
   exists, but there is no keyboard path through the whole surface and no
   phone layout for the conversation.

## What must not be lost

Every tab in the manifest above, the capability gating that disables a tab
with a refusal, the swap proxy switch, the operations shell with recovery
and cancel, the analysis inspectors with their exactness refusals, the API
key handling with opt-in persistence, the same-origin policy with no
external assets, and the real-browser acceptance lane. The campaign keeps
`app.html` and its modules serving unchanged behind a Workbench entry until
a front-page acceptance step proves an absorbed capability.

## Server contracts the front page will use

| Route | Use |
| --- | --- |
| `/health`, `/health?swap=<name>` | Liveness; live model switch through the proxy |
| `/catalog/models` | Servable catalog with presence, recipe, staleness, capabilities and coverage |
| `/analyze/model`, `/props` | Served model identity, context length, default generation settings |
| `/workspace/manifest` | Capability-gated tab declarations; the front page derives composer modes from the same gating |
| `/v1/chat/completions` (+`/input_tokens`) | Streamed conversation with protocol content parts |
| `/v1/images/generations`, `/v1/videos/generations`, `/v1/videos/edits`, `/v1/audio/speech` | Generation modes |
| `/v1/embeddings`, `/v1/rerank` | Embedding and reranking modes |
| `/analyze/logits`, `/analyze/states`, `/analyze/attention`, `/analyze/vocab`, `/analyze/tensors` | Turn inspection |
| `/operations/evidence` | Operations strip |
| `/agents`, `/automations`, `/peers` | Retained workbench surfaces |

Interaction-record endpoints for listing and resuming conversations are the
one contract the front page needs that the present client does not call;
the durable-sessions row binds them.

## Appendix: go-micro pattern review (2026-09-04)

Reviewed `C:/Users/jeffm/go-micro` (go-micro.dev/v6, an agent harness and
service framework; 661 Go files, a small server-rendered dashboard under
`cmd/micro/web` with home, agent playground, API explorer, logs, status and
auth pages). Nothing is copied or imported from that tree; the items below
are behaviors and contracts the campaign adopts, mapped to the rows that
carry them. Overgo keeps UTCP tool manuals, its own store and its own
interaction records; MCP, A2A and payments are out of scope.

| Pattern seen | What it is there | Where it lands here |
| --- | --- | --- |
| Typed stream events | An agent stream is `tool_start`, `tool_end`, `token`, `done` events; the UI renders tool calls as collapsible cards with input, result, status and elapsed time, a thinking row, and errors as rows | `gui-simplify/one-composer`: one event vocabulary (token, tool_start, tool_end, media, usage, done, error) as the server's documented streaming contract; one renderer |
| Checkpointed run resume | A run is a durable record of steps; resuming replays completed parts from the record and continues the rest without re-executing finished tool calls | `gui-conversations/durable-sessions`: an in-flight turn survives reload or disconnect by reattaching to its run record |
| Observable memory compaction | Older turns compact into a deterministic summary, recent turns stay verbatim, the summary is inspectable | `gui-conversations/durable-sessions`: context meter, compaction and context-shift events visible in the thread, summary inspectable |
| Settings held by the server | Provider, model and key are saved server-side and reloaded by the page | `gui-conversations/durable-sessions`: generation settings and system prompt live in the interaction record |
| Empty state as the getting-started card | The playground's empty state names the available tools and gives three numbered steps; settings open by themselves when nothing is configured | `gui-shell/front-page`: the empty state shows the served model, evidence, capabilities and three starter actions; a setup card when nothing is served |
| Guardrails and human-in-the-loop | `MaxSteps`, `LoopLimit`, `ApproveTool` (a policy hook before each tool call); tool middleware wraps every call | `gui-workbench/agent-in-thread`: guardrails visible with what remains; approvals as inline cards bound to the inbox record; tool calls as cards |
| Plan and delegate as tools | The agent records a plan in memory and delegates subtasks to another agent or a short-lived sub-agent that cannot re-delegate | `gui-workbench/agent-in-thread`: a recorded plan renders as a live checklist; a delegated sub-session as a nested card with its own receipt |
| Run inspection with a status vocabulary | `micro inspect` lists runs by status (running, done, canceled, timeout, refused, error, ...) with trace ids, for people and for automation (`--json`) | `gui-workbench/inspect-turn`: the turn's run record with one declared status vocabulary, timings, receipt and trace identities |
| First-run flow as a verified contract | "0 to 1" and "0 to hero" walkthroughs are tests that run on every change | `gui-quality/acceptance-lane`: the first-run flow is a browser-lane walkthrough selected by the code manifest for every webui commit |
| Status and logs pages | A status dot, service list with running and stopped counts, per-service live log view | `gui-workbench/operations-strip`: header status dots for server, proxy and device; live log tail of the running operation |
| API explorer from the registry | Endpoints listed from the registry with generated forms that submit and show the response | `gui-simplify/one-route-table`: an explorer rendered from the route table and the API manifest with schema-driven forms |
| Hot reload in development | `micro run` watches files and rebuilds | `gui-simplify/one-shell`: a server flag serves webui assets from disk with caching off |
| CLI-first, UI must earn its place | Roadmap principle: the CLI is the experience; UI is added only where it carries its weight | Campaign doctrine: every GUI surface is a projection of a record the CLI and the store already expose; nothing is reachable from the browser that is not equally reachable from the control plane |

Not adopted: server-rendered Go templates (Overgo's client is a thin static
client over JSON, which the acceptance lane already drives), the multi-user
auth pages (single operator, bearer key), MCP and A2A gateways (UTCP is the
tool contract), and paid tools.
