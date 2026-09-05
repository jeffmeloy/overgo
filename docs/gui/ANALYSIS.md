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

## Closeout and merge readiness (2026-09-05)

The campaign's rows landed on the lane branch `professional_overgo_gui`
(fork from master at `0c1a9170`; merge-base with master `9dce4fca`; at
closeout the lane is 17 commits ahead of master and 7 behind). The
simplification report is [SIMPLIFICATION.md](SIMPLIFICATION.md), generated
by `go run ./cmd/webui-lane -report` from the fork's tree and the head's:
the client census at both, beside the behaviours the acceptance lane proved.

Both acceptance lanes pass on the lane head: the workbench acceptance steps,
the front page's keyboard, motion, colour and width contract, and the
first-run journey against a served model through the real swap proxy
(Qwen2.5-0.5B served, the image leg proven as the declared refusal, the
switch through the picker taken to the next servable model). The gate's
webui-lane check runs the same lane for every web UI change.

Every finding parked during the campaign carries a disposition. Three are
deferred with their closure path recorded: agent mode in the conversation
has no loop bound, plan checklist or delegated cards (no server record
declares them); the journey's vision leg proves the refusal branch only (no
servable vision-capable model in the store); the gate's test phase once
failed on a Windows process-start status from a nested `go test` spawn under
parallel load (both tests pass alone; the relaunched gate passed).

Merge protocol for the master worktree (see the lane merge protocol):

- The lane's `docs/plan.json` never merges: master's plan wins (resolve with
  `--ours`); the merge commit binds to a master plan row.
- Merge with `git merge --no-ff --no-commit professional_overgo_gui`, then
  `go run ./cmd/gate -merge -plan <item>/<step> -message-file <f>`. Without a
  prepared common ancestor use `-plan-projection first-parent-target
  -merge-source-store C:/Users/jeffm/professional_overgo_gui/overgodb-store`.
- Import the lane's closure store before gating
  (`closure-scan -import-store C:/Users/jeffm/professional_overgo_gui/overgodb-store`),
  then self-import; publish the modern-Go census last
  (`modern-census -publish-census`); rebuild `bin/gate.exe` after the merge,
  since the repoanalysis authority tables changed (the cmd/recipe process
  allowance and its entry-authority exception were removed when the intake
  steps moved to `internal/modelintake`).
- What master absorbs: `internal/modelintake` (extracted from cmd/recipe),
  the `/library/register` and `/library/validate` routes with their workspace
  audit allowance, the internal/server internal-imports budget at 41, the
  API manifest, SBOM, modern-Go baseline and census, the front page and its
  composer, the lane packages `internal/webuilane` and `cmd/webui-lane`, and
  the gate's webui-lane check.
- Verification after the merge: `go build ./...`,
  `go test ./internal/server -run 'TestWebUI|TestFrontPage' -count=1`, and
  `go run ./cmd/webui-lane` against a store holding a servable model.

## Multimodal in and out (2026-09-05, owner directive)

The owner's directive after the closeout: the GUI is multimodal in and out
for every model and modality. Seven rows landed on the same lane branch
after the closeout commit (`9d85697c` serve-projectors, `47af2db5`
register-projectors, `3230c35c` library fixes, `788eea20` generation
workspace, `e0d870f2` generation declarations, `e785c516` media roundtrip,
`cfe9a4a2` journey modalities), and this closeout regenerates the
simplification report over the whole lane: the JavaScript ceiling stands
at 4728, the three generation tabs are gone, and the client names 81
manifest routes.

What the rows established, each a record rather than a statement:

- The server resolves a served model's projector from its active
  projection recipe when no `-mmproj` is given (`discovery.ActiveProjector`),
  so the swap proxy, `overgo_gui.bat` and the lane serve image, audio and
  video input for every model that declares it. The capability catalog
  pages the alias projection to its end; one bounded page had hidden the
  27B's inference activation.
- The Library registers a projector beside or with a model (files sorted by
  their own metadata), validates it with the model (the projector opens on
  the host and declares its media), and activates the projection. The lane
  store holds projection activations for Qwen3.8-27B, Qwen3.5-4B and
  gemma-4-12B-it-fp8-native, the last two made through the route. The
  Gemma 4 E4B projector converts to the tower form whose catalog descriptor
  declares no media; that is a parked finding for `internal/projector`.
- A generation workspace over the store (`internal/server/generation_store_workspace.go`)
  lists every active media recipe with the controls its request type
  declares (`internal/mediacapability`, the catalog extracted from
  cmd/recipe) and executes through the same executors, publishing PNG, GIF
  and WAV artifacts behind run records. Wan runs from the page's prompt
  form (`internal/latentvideo/wan_form.go`): a prompt, a negative prompt and
  a seed are typed controls, the runtime derives the conditioning contexts
  through the model's own umT5 encoder and the noise plan from the seed on
  its device, and a generation parameter left blank takes the profile's
  value. LiveEdit runs from the clip form (`internal/latentvideo/edit_form.go`):
  a prompt, a seed and a source clip's artifact id, resolved by the video
  capability before the mapped session director into the compiled
  condition (the declared edit schedule in `edit_profiles.json`, the text
  context through the Wan text pipeline, flow sigmas for its timesteps)
  and the planar source video decoded from the clip's GIF; a clip whose
  latent frames the schedule cannot chunk is refused with the admissible
  counts. The clip reference is a typed artifact id: the controls
  reflection declares it as an artifact control, and a media card's "use as
  input" fills it with the card's stored id when the chosen mode's request
  names one (the journey's clip legs prove it on the oscillator video and
  LiveEdit). The HTTP package names neither the media executor catalog
  nor the model intake: the launcher binds `GenerationCatalog` (executors,
  declared controls, output content) and `LibraryIntake`
  (`internal/libraryintake`: model files, register, validate) into the
  server, which keeps its internal-import budget at 42 after the audio
  merge. The capability runtime replays an
  identical request from its recorded artifact, so media output types
  decode their own recorded content. The VQA activation runs from the page
  (`internal/vqaserve`, lifted from cmd/vqaparity's serving copies: the
  processor, the device pipeline of vision tower, merger, chained prefill
  and decode, and the two-stage recipe execution; the harness keeps its
  goldens and verifiers and drives the shared pipeline with the golden
  chain as the per-step verifier). The catalog's vqa capability takes a
  stored image artifact, a question and an optional decode budget (the
  serving declaration `serve_policy.json` when blank, the budget the
  activation evidence is produced under), reads the image from the store,
  and returns the answer text, which the workspace publishes as a
  plain-text output artifact; the page's vqa mode feeds the message body to
  the question control, takes the image from a media card's "use as input",
  and renders the answer as the assistant's turn. `TestVQAActivation`
  proves the canonical RxBrain case through the catalog's executor on the
  device.
- The front page renders every generation mode from the declaration (models
  by name, refusals, controls, exported choices such as voices), runs
  through `/generation/run` and the operation wait, and every media card
  offers itself back as input.
- The journey proves image in on Qwen3.5-4B (switched to by declaration),
  image out of Un-0, the image back in and refused by Qwen2.5-0.5B with its
  declared reason, and a speech clip out of pocket-tts with an exported
  voice, in one real browser session.

Additional facts for the master merge:

- `internal/mediacapability` replaces cmd/recipe's capability files (the
  compatibility claim `sensenova-production-image-oracle` names the moved
  evidence path); the repoanalysis authority table declares the three
  executor call sites; the internal/server import budget is 42; the clone
  ceiling is 9999 under the review recorded in 788eea20.
- `internal/dataroot` resolves the `OVERGO_DATA_ROOT` base as a working
  directory of its own, so a base whose `local-models.json` points at data
  in another home resolves from any tree; the lane's `build/latentvideo`
  holds the retained Wan fixtures the moved video test reads.
- The gate's command runner relaunches a child Windows could not start
  (STATUS_DLL_INIT_FAILED), three attempts two seconds apart; the deferred
  finding for that flake records the fix in 788eea20 (the finding tool
  disposes once, so its record stays deferred). The journey's vision
  finding likewise stays deferred by record while its closure landed in
  cfe9a4a2.
- The runtime event stream reconnects while anyone listens and an operation
  wait falls back to `/operations/wait` when the stream breaks under a swap.
