# Image and video generation report

This report covers generation validation, resource use, and targeted optimization
in `codex/image_video_gen`. The worktree starts from master commit `a5798b52`.
[docs/plan.json](docs/plan.json) defines the execution steps and acceptance checks.

**Status: isolated storage, readiness, and inventory checks passed.** No
image/video generation or optimization comparison has run in this worktree.
The entries below identify existing implementations and the evidence the
campaign must collect or verify. They do not claim new model results.

Quality protocol acceptance passed and committed as `d79294cc`. It validates
case definitions and evidence bindings, while retaining the historical output
failures described below. Resource protocol checks also pass locally; their
gate remains pending.

## Scope

Validate existing image and video generation recipes, including conditioning and
editing where the active recipe declares support. Compare outputs with pinned
references, inspect visual and temporal quality, measure complete generation
cost, and optimize measured bottlenecks while preserving required behavior.
Prioritize common code components, processing speed, CPU/GPU memory reduction,
and fewer transfers between host and device memory.

Use the worktree's private evidence store for all new records. Reuse inherited
results only after checking their model, recipe, source, environment, and request
identities. Keep model weights read-only at their declared locations.

Speech, transcription, VQA, model training, and unrelated text benchmarks remain
outside this campaign. Model names alone do not establish a supported task.

## Frozen coverage inventory

The [inventory specification](docs/image_video_inventory.json) binds the seven
active model/task/recipe entries, reference-file hashes, and known limitations.
The existing census owner published snapshot
`evidence:sha256:42ce9a29b26232e6f68937df366a4f26c1f81199cca38198c10bd2c89b9fcb45`
from source commit `476ca45918edbc447a00cb2fca628755d38edfd5`, at private-store
sequence 26673. The census includes the registered catalog; this campaign uses
only its image-generation, video-generation, and video-edit projection.

| Model | Active task | Existing implementation | Remaining validation |
| --- | --- | --- | --- |
| Krea-2-Turbo | Image generation | [Latent image runtime](internal/latentimage) | Matched quality and resource protocol; incomplete intermediate golden captures |
| SimpleDiffusion-TensorProductAttentionRope | Image generation | [Diffusion image runtime](internal/diffusionimage) | Numerical reference and perceptual-quality checks remain separate |
| SenseNova-U1-8B-MoT-Infographic-V3 | Image generation | [Image recipe integration](internal/mediacapability/image_cuda_windows.go) | Separate four-step smoke and full-resolution reference requests |
| Un-0 | Class-conditioned image and video generation | [Oscillator runtime](internal/oscillatorimage) | Native sample parity, declared geometry, and video timing |
| Wan2.1-T2V-1.3B | Video generation | [Latent video runtime](internal/latentvideo), [runner](cmd/latentvideo-run) | Numerical trajectories, full-clip quality, and resource measurements |
| LiveEdit | Source-conditioned video generation | [Media recipe integration](internal/mediacapability), [latent video runtime](internal/latentvideo) | Full-quality source-conditioned oracle and video timing |
| Declared compositions | Composition review remains open | [Composition validation](cmd/composite-generation-lane) | Bind supported component recipes and input/output contracts |

Record candidate-only implementations separately from activated recipes. Add or
remove discovery entries only after inspecting the catalog; retain the reason.
Mark missing required evidence as unavailable or incomplete, never as passed.

The store also contains four superseded media definitions: prior Un-0 image,
SimpleDiffusion image, Wan video, and Krea image recipes. Un-0 video recipe
`514d3feb338d82271a67ac4b52c7281378ab300b478ba17bd2b338be497b6761` remains a
candidate with host placement; its lifecycle has not reached verification and
activation. These five definitions do not add active capability entries.

## Worktree readiness

The existing `cmd/overgodb-backup` copied and byte-verified 77,508 files
(2,260,357,507 bytes) in 9 minutes 34 seconds. Replay matched imported sequence
26279 and head
`131030a35385ba8a46e7eba525befdff8a06793cebf32ccfb2bebe4a2980356c`.
The source was `C:/Users/jeffm/overgo/overgodb-store`; the independent writable
copy is `C:/Users/jeffm/image_video_gen/overgodb-store`.

The ignored `local-models.json` references existing model files under
`C:/Users/jeffm/adaptive_new/models` and datasets under
`C:/Users/jeffm/overgo/datasets`. Checkpoints resolve inside this worktree.
Run campaign commands with `OVERGO_DATA_ROOT=C:/Users/jeffm/image_video_gen`.
The existing media resolver found FFmpeg at
`C:/Program Files/DownloadHelper CoApp/ffmpeg.exe`.

The local configuration also uses the existing `audio_reference_store` setting
to locate inherited regression fixtures beside the original store. Those checks
read the existing models and datasets and write to temporary test stores.

The readiness acceptance verifies the imported commit and inherited census,
checks independent store files and data roots, and validates embedded kernel
assets. Its fixtures reject missing roots, an external store, a mismatched
snapshot, missing evidence, and hardlinked files; a valid local append retains
the imported identity. These checks execute no model.

Dispatch initially rejected the two local planning commits because they removed
unfinished plan identities. They remain recoverable on
`codex/image_video_gen-planning-backup`. The corrected plan retains inherited
obligations under `master` or their existing `operator` owner, and assigns the
media steps to `image_video_gen`. This lane does not execute inherited master
work. Its final step is `media-closeout/closeout`.

Two gate attempts took 828.1 and 870.9 seconds. The first exposed missing
audio-reference paths and a package-wide test timeout. The configured reference
root resolved the path failures; both affected fixture tests passed. The second
attempt failed only because the gate package exceeded Go's default ten-minute
aggregate timeout while other tests remained paused or unfinished.

This worktree incorporates the relevant active-test deadline implementation and
caller updates from master commit `bd3d1dc1`. It applies Go's default bound to
each top-level test's active time and event silence, excludes paused time, and
preserves explicit operator timeouts. The accompanying deadline and caller
tests remain required. This prerequisite does not change generation behavior.

The next gate passed in 1,142.3 seconds and committed readiness as `476ca459`.
It executed 230 package checks with no reused package receipts. Owner tests took
7 minutes 41 seconds, dependent tests 8 minutes 4 seconds, and browser checks
1 minute 58 seconds; these phases overlap, so their sum is not elapsed time.

The inventory gate passed in 1,031.7 seconds and committed as `5eeba8ba`.
It executed 230 package checks with no reused package receipts. Concurrent
work continued in the master worktree; these gate durations are validation
cost observations, not isolated generation benchmarks.

The quality gate passed in 1,014.4 seconds, again executing 230 package checks
with no reused receipts. A preceding admission attempt refused three outdated
JSON omission tags; those tags now follow the repository's Go 1.26 convention.

## Retained evidence inspection

A read-only query of the private snapshot found seven active image/video
capabilities across six models. All seven resolved their model files and
activation evidence without a stale status. This establishes the catalog state;
the inventory acceptance passed, while quality review remains open.

| Model | Declared task | Inputs | Retained runs with output artifacts |
| --- | --- | --- | ---: |
| Krea-2-Turbo | Image generation | Conditioning | 1 |
| SimpleDiffusion-TensorProductAttentionRope | Image generation | Conditioning | 2 |
| SenseNova-U1-8B-MoT-Infographic-V3 | Image generation | Conditioning | 3 |
| Un-0 | Image generation | Conditioning | 244 |
| Un-0 | Video generation | Conditioning | 162 |
| Wan2.1-T2V-1.3B | Video generation | Conditioning | 3 |
| LiveEdit | Video generation | Conditioning and source | 2 |

The catalog contains no active `video-edit` recipe. LiveEdit's active recipe
declares source-conditioned video generation. Its name does not expand that
contract.

All 16 retained PNG/GIF exports decode and match their SHA-256 filenames.
The Wan manifests `g3_denoise.json` and `g4_denoise.json` reference 110 and 16
raw float32 assets respectively; the existing fixture loader verified their
hashes and declared element counts. These are artifact checks, not new model
comparisons.

Decoded GIF timing exposes accumulated truncation in the shared
`media.GIFFrameDelay` helper. The 81-frame Wan sample stores six centiseconds per
frame: 4.86 seconds, compared with 5.0625 seconds at its declared 16 fps. The
12-frame Un-0 samples store twelve centiseconds per frame: 1.44 seconds, compared
with 1.5 seconds at 8 fps. A timing correction must preserve frame pixels and
order, respect GIF time resolution, and retain the original artifacts. Their
existing encoded-byte reference checks also need an explicit disposition before
changing the encoding policy.

The 417 retained successful generation records lack source revision, environment,
and elapsed-time fields; 389 retain input artifacts. Separate activation verifier
records contain source identity and timings, but their outputs contain verification
evidence rather than the generated image or clip. The campaign must establish an
execution-specific link before combining these records in a performance result.
Retained samples can inform case selection after their inputs and output content
are checked.

The shared workflow runtime returns per-node timing separately from its persisted
generation record. Capability verification publishes measured records through
`internal/modelintake`, which does not currently bind the measured request or
generated output to that verification record. These existing owners are the
first places to address the measurement gap.

Source inspection also identified a transfer candidate in Wan's denoiser session:
each forward step uploads host inputs and reads the output head back to the host.
The latent-image and diffusion-image implementations already use retained device
storage. Profiling must establish which boundaries can share device buffers while
preserving each model's sampler and tensor-lifetime requirements. No transfer or
memory improvement has been measured yet.

## Validation requirements

| Area | Required evidence |
| --- | --- |
| Identity and reproducibility | Model/component hashes, recipe and sampling policy, source and reference revisions, environment, exact prompt/input artifacts, seed, dimensions, precision, and frame settings |
| Image correctness | Successful decode, expected dimensions/channel order, finite values, declared numerical reference comparisons, and input-to-output lineage |
| Image quality | Fixed review cases for prompt adherence, conditioning preservation, composition, and visible artifacts; justified quantitative measures where applicable |
| Video correctness | Expected frame count, order, timing, duration, spatial dimensions, causal decoding, and supplied clip/conditioning preservation |
| Video quality | Fixed review cases for motion, flicker, temporal consistency, requested content, and declared editing behavior |
| Reliability | Cancellation and error handling during each phase, resource release, bounded repeated use, retained completed outputs, and explicit restart/resume behavior |
| Public workflows | API request/response behavior, declared controls, progress, cancellation, output retrieval, downloads, and browser evidence where required |

Freeze criteria and reference inputs before comparing an implementation change.
Keep functional, numerical, visual, performance, and interaction results separate.
Store negative results and evaluate the complete required case set.

The [quality protocol](docs/image_video_protocol.json) freezes 15 retained
requests across all seven active recipe entries. Each case identifies its input
artifacts, source run, output artifacts, decoded geometry, and applicable video
timing. The protocol identifies the existing numerical test owners, preserves
their reference criteria, and distinguishes smoke requests from full visual
review. Its acceptance checks exact lineage and rejects substituted requests,
changed dimensions, omitted cases, and changed review criteria.

The review asks whether the output contains the requested subject or class,
preserves required source content, and has visible defects in composition,
detail, text, or color. Video review covers the complete frame sequence and
records motion, discontinuities, flicker, and consistency. Un-0 generates
independent class-conditioned frames with successive seeds; this behavior
differs from the temporal conditioning used by video models.

All seven retained video cases fail the protocol's duration requirement. GIF
represents time in centiseconds, so accumulated duration error must stay within
half a centisecond when rounded to the nearest representable duration. These
existing failures remain visible. Passing protocol acceptance establishes the
case definitions and checks; it does not mark their model outputs as correct.

## Compute environment and measurements

Initial host discovery on 2026-09-12 found an AMD Ryzen 9 9950X with 32 logical
threads, 93.6 GiB of visible RAM, and an NVIDIA RTX 4090 D with 48.0 GiB VRAM,
114 multiprocessors, and compute capability 8.9. The CUDA driver reports version
13.4; the installed driver is 616.64. About 59.3 GiB RAM and 47.5 GiB VRAM were
free at discovery. These observations do not reserve capacity for later work.
The existing CUDA metadata command and kernel-manifest validation passed.

Recheck usable RAM, per-device VRAM, CPU/GPU capacity, driver/kernel versions,
disk capacity, and concurrent reservations at admission.
Use existing resource admission and supported execution policies. Derive batching,
tiling, and concurrency from the workload and available capacity.

For each model and request, record:

- Cold-start and warm-run elapsed time, with preparation and warmup identified.
- Model load, conditioning/encode, denoise/integrate, decode, export, and cleanup time.
- Host-to-device, device-to-host, and device-to-device transfer bytes and call counts,
  classified by weights, conditioning, latents, intermediates, and final outputs.
- Transfer/compute waiting, synchronization, CPU conversion and staging costs.
- Active and cached allocations, buffer lifetimes, changing-shape reuse, host and
  pinned RAM, and process/device peaks as distinct measurements.
- Output size, frame count/rate, and throughput under the same quality requirements.
- Waiting, failed attempts, verification, and publication cost in the full comparison.

Choose repetitions and stopping criteria from observed variability and declared
measurement needs. Record unmeasured quantities explicitly. Do not report summed
parallel phase durations as total elapsed time.

The [resource protocol](docs/image_video_resources.json) binds these requirements
to the frozen quality criteria. Its acceptance reads current host RAM, CPU load,
disk capacity, GPU identity and free VRAM without reserving a device. It exercises
the existing capacity and budget owners, rejects omitted measurement requirements,
and preserves the difference between an observed zero and an unknown value.
Separate owner tests verify shared/exclusive admission and process-exit cleanup.

Use the existing `processmeasure` owner for process working-set peaks and the
CUDA driver's `MemoryStats` and `ExecutionStats` for scoped allocation and copy
observations. The common resource record does not currently carry every driver
counter or retained-memory scope; preserve those details alongside it. The user
specified no additional numeric ceiling. Existing command deadlines still apply,
and each acquisition group must fit a declared workload and current capacity.

## Optimization comparisons

No optimization comparison has run in this worktree. Review existing improvements
before proposing new work: the resident diffusion sampler retains GPU state,
and decoder cleanup releases temporary outputs after cancellation.

Each proposed change must identify its existing implementation owner, measured
cost, expected effect, unchanged acceptance criteria, and rollback. Compare
baseline and candidate on matched artifacts, requests, hardware, and protocols.
Start with retained evidence and a small baseline set that exercises the actual
shared image/video components. Collect quality, timing, memory, transfer, and
sample evidence together where measurement requirements permit. Reuse these
results in the full model validation; do not wait for unrelated models before
testing a bounded improvement.

| Optimization area | Planned work and acceptance |
| --- | --- |
| Common components | Consolidate equivalent preparation, retained-output handoff, conversion, or cleanup in existing owners. Migrate real callers, delete duplicate implementations, and preserve intentional model differences. |
| CPU/GPU transfers | Retain compatible intermediate tensors on the GPU, reuse weights and conditioning, eliminate unnecessary round trips, and combine required small transfers. Measure bytes, calls, waits, and complete generation time. |
| Memory use | Remove duplicate copies, shorten live allocations, reuse existing scratch pools, and use supported incremental decode or tiling. Measure host and device peaks, retained storage, and repeated-request behavior. |
| Processing speed | Reuse preparation and compiled graphs, remove redundant conversion and synchronization, and consider fusion or overlap only where measured cost and numerical requirements justify them. |
| Combined behavior | Compare accepted changes together across affected image/video consumers and supported capacity profiles. Preserve quality, cancellation, request isolation, and declared resource limits. |

Use [resident graph sessions](internal/graphruntime/resident.go),
[executor buffer pools](internal/cuda/executor/device_buffer_pool.go), existing
device/driver operations, and the current image/video runtimes. Add shared
behavior where real callers need it; do not add a second allocator, transfer
scheduler, or media engine. Include owning-component tests as well as model-level
comparisons when changing shared code.

Retain buffers until their final consumer completes. If measurements justify
pinned memory or asynchronous transfer, bound host storage and verify stream
ordering, cancellation, and release after transfer completion. Passing a device
pointer alone does not establish compatible layout, lifetime, or device ownership.

Report tradeoffs explicitly. Keeping tensors on the GPU can reduce transfer time
while increasing retained VRAM; freeing or recomputing them can reduce memory
while increasing latency. Adopt changes against the declared quality, speed,
and memory requirements, using the actual compute environment.

| Required comparison field | Publication requirement |
| --- | --- |
| Change and causal hypothesis | Identify the code change and the measured cost it should reduce |
| Common-code refactoring | Identify shared owners, migrated callers, deleted duplicate paths, and added code |
| Baseline and candidate | Record exact source, model, recipe, request, and environment identities |
| Quality and reliability | Preserve the required numerical, visual, temporal, and lifecycle checks |
| Cost | Compare cold/warm latency, active/retained/peak host and device memory, transfer bytes/calls, synchronization, and total experiment cost |
| Decision | Accept, reject, or retain further work with the recorded reason and evidence |

## Image and video samples

No samples have been generated or exported by this worktree yet. Export actual
stored outputs through the existing compatibility/sample code into
`docs/media_samples`. Show representative baseline/candidate images, playable
clips, and video frames from the frozen case set, including material failures.

For every sample, retain the prompt or conditioning artifact, seed, model/recipe,
source commit, environment, dimensions/frame settings, output hash, and run/result
identities. Identify thumbnails and transcodes as derivatives and link the original.

## Execution and reproducibility

The plan prepares the private data root and freezes coverage, quality, and
resource requirements. It maps common components and data movement, selects a
representative baseline, and evaluates refactoring, transfer, memory, and processing
changes. Independent model validation and report preparation can proceed when
their prerequisites are ready. Serialize edits to the same component and exclusive
measurements through existing ownership controls.

Export samples as model bundles complete. The final checks reconcile affected
consumer results, combined optimization behavior, constrained-capacity behavior,
lifecycle and public workflows, and report completeness without repeating valid
generation work.

Existing execution paths include `cmd/recipe verify`, `cmd/recipe run`,
`cmd/latentvideo-run`, and `cmd/composite-generation-lane`. Resolve each command's
actual recipe, store, inputs, and resource requirements before execution. Use
`cmd/device-lane` and `cmd/webui-lane` for the applicable checks; direct tests that
skip required device/browser work do not satisfy acceptance.

The existing generator supports this focused report with
`cmd/compatibility -update-media -media-scope image-video` and exports its frozen
review requests with `-export-samples -media-scope image-video`. The default
scope retains the all-media report. Both use the same run and sample readers.
The focused projection includes source, environment, input, run and output
identities, preserves failed and cancelled attempts, and identifies missing
measurements. Export and rendering check sample hashes and image/clip decoding.
The final publication step will replace this working report with the completed
evidence projection.

## Remaining work

The [plan](docs/plan.json) records remaining steps and their dependencies.
The active inventory is reconciled with the private snapshot. Quality and resource
requirements are frozen before new measurements. Implement each named
acceptance check before its corresponding acquisition.
Model quality, performance, lifecycle, and workflow validation remain open.
