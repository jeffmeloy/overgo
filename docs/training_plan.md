# Training Plan — per-model, memory-hierarchy-aware

Single-box target (measured 2026-08-10): **VRAM 48 GB** (RTX 4090D, ~1 TB/s),
**CPU RAM 94 GB** (PCIe4 x16 ~25 GB/s to GPU, ~80 GB/s local), **NVMe ~1.1 TB
free** (Gen4 ~6 GB/s). Fastest memory does the math; slower tiers are staging.

## 1. Training-state memory model

Per P parameters, with Muon (momentum-only — no second moment, the key win over
Adam):

| Component | bytes/param | P=1B |
|---|---|---|
| Weights (bf16) | 2 | 2 GB |
| Gradients (bf16) | 2 | 2 GB |
| Muon momentum (fp32) | 4 | 4 GB |
| **Optimizer+model state** | **8** | **8 GB** |
| Activations (per micro-batch, checkpointed) | ~O(1)/segment | batch-dependent |

Adam would be 12–16 B/param; Muon's single fp32 momentum is what makes 8 B/param
achievable. Momentum may drop to bf16 (→6 B/param) with periodic fp32 refresh, or
offload to NVMe. Newton-Schulz needs only the *current* 2D group's matrix + its
Gram on-device transiently.

State budget vs the 48 GB card: everything ≤ ~5B params trains fully resident;
≥6B needs tiered offload.

## 2. Tiered-offload architecture (ZeRO-Infinity-style, single-GPU)

- **Tier 0 — VRAM (compute):** active layer weights + its activations + the
  current Newton-Schulz matrix + a double-buffer for the next layer's weights
  (prefetch) + the current micro-batch's live (non-checkpointed) activations.
- **Tier 1 — CPU RAM (primary offload):** full resident bf16 weights, the fp32
  momentum, the gradient-accumulation buffer, spilled checkpointed activations.
  94 GB holds the *entire* state of every trainable model here except the 12B.
- **Tier 2 — NVMe (cold overflow):** fp32 master/optimizer state that exceeds
  CPU RAM (only the 12B), dataset shards (streamed), activation-spill overflow.
  Async, double-buffered.

### Scheduling primitives
1. **Layer-streaming + double-buffered prefetch.** Compute layer N in VRAM while
   DMA-prefetching layer N+1 weights CPU→VRAM; offload layer N grads VRAM→CPU
   after its backward. Reuses the retained-graph replay + device-feed machinery
   already in the executor.
2. **Break-even rule (the intelligence).** A layer with W weight-bytes and C
   FLOPs at micro-batch B: fetch ≈ W/25e9 s, compute ≈ C/165e12 s (bf16). Offload
   is *free* (compute-bound, PCIe fully hidden) when `C/165e12 > W/25e9`. Since C
   scales with B and W does not, **there is a minimum micro-batch × seq that makes
   offload free** — the scheduler picks B (or grad-accum depth) at or above it;
   below it, keep more layers resident instead of offloading.
3. **Gradient checkpointing.** Store only segment boundaries; recompute
   activations in backward. Activation memory O(layers)→O(√layers) or O(1)/segment
   at ~33% extra compute — mandatory for ≥4B.
4. **Gradient accumulation.** Decouple statistical batch from memory batch:
   micro-batch sized to VRAM, accumulate K steps (buffer in Tier 1) to the target
   effective batch.
5. **Per-group Muon streaming.** The optimizer step streams each 2D param-group
   through VRAM: momentum (Tier 1) + matrix DMA-in → Newton-Schulz → DMA-out. Only
   one group's matrix is on-device at a time → device Muon at any scale without
   holding all optimizer state in VRAM. (Host NS is prohibitive ≥500M — this is
   how Muon scales; a device NS kernel is the enabling to-do.)

## 2b. Dynamic quantization tier (amplifies offload)

overgo ships a full quantization codec both directions (`internal/quant`:
`quantize.go`, calibrated `quantize_weighted.go`, `dequantize.go`) over Q1_0–Q8_K,
MXFP4, NVFP4, F8E4M3, BF16/F16, with in-kernel dequant already proven on-device
(native-fp8 matmul, GGML block dequant). Quantization is therefore a *tier
compressor*, applied per-slot and chosen dynamically by the break-even rule, not a
fixed model format.

**What gets quantized, and why it stays correct:**
- **Offloaded/streamed forward weights (Tier 1/2 → VRAM):** store the resident
  copy quantized (Q6_K ≈ 6.5 b/p, Q4_K ≈ 4.5, NVFP4/MXFP4 ≈ 4, fp8 = 8) and
  dequant in-VRAM before the matmul. This shrinks both the resident footprint *and*
  the PCIe/NVMe transfer → **offload goes compute-bound at a smaller micro-batch**
  (W drops, so the §2 break-even `C/165e12 > W/25e9` holds sooner). fp8 uses the
  existing native matmul (no separate dequant pass).
- **The master weight is NOT quantized.** Keep a bf16 (or fp32) master +
  momentum for the *update*; quantize only the forward compute copy. The gradient
  is taken w.r.t. the dequantized forward weight; the optimizer update applies to
  the master. So quant buys bandwidth/footprint without corrupting convergence.
  (QLoRA-style frozen-quant base + trained hi-precision adapter is the alternative
  when only a delta is learned — cheaper still, but not full fine-tune.)
- **Gradients transferred VRAM→CPU:** quantize to fp8/Q8 on the way out,
  dequant on accumulation → halves–quarters the backward PCIe traffic.
- **Optimizer momentum (Tier 1/2):** 8-bit / bf16 momentum with per-tensor
  scale via `quantize_weighted` — the "8-bit optimizer" — cuts the largest state
  buffer. This is what lets the 12B momentum fit CPU RAM without NVMe.
- **Activation spill (VRAM→CPU):** Q8/fp8 the checkpointed activations that spill.

**Dynamic selection (the intelligence):** per layer, start at the highest
precision that is compute-bound; if the offload is transfer-bound at the target
batch, step the *offloaded/transferred* precision down (Q6_K→Q4_K→NVFP4→fp8) until
compute-bound or the quality gate binds — never the master. Every choice is a
recorded decision with a **measured** grad-parity/loss-decrease bound (quant of
the forward weight perturbs the gradient; it must stay within the FD-oracle
tolerance and keep loss monotone), not a magic constant.

## 3. Per-model plans

Legend: **R**=fully resident (Tier 0 only), **T1**=CPU-RAM offload, **T2**=+NVMe.
Effective batch reached via grad-accum. Only models adaptive actually
trains (or that have an overgo train lane) get a training plan; the rest are
serving/generation-only or pretrained (marked N/A).

| Model | P | State (8B/p) | Tier | Micro-batch policy | Ckpt | Fit |
|---|---|---|---|---|---|---|
| Fractale-350M | 0.39B | ~3 GB | **R** | large (seq 2k, B up to VRAM) | off | trivial; ~40 GB free for activations |
| Carbon-500M | 0.5B | 4 GB | **R** | large | off | trivial |
| Qwen2.5-0.5B | 0.5B | 4 GB | **R** | large | off | trivial |
| SimpleDiffusion / Un-0 / pocket-tts | <1B | <8 GB | **R** | family train.go step-verified | off | resident |
| MiniCPM5-1B | 1B | 8 GB | **R** | B sized to fill ~35 GB activations | opt | resident, big batch |
| E4B (gemma3n) | ~8B | 64 GB | **T1** | micro-batch 1–2, grad-accum to target; layer-stream W/G | **on** | 64 GB in 94 GB CPU RAM; active window + ckpt acts in VRAM |
| gemma-12B (gemma4_unified) | 11.96B | 96 GB | **T1+T2** | micro-batch 1, grad-accum; bf16 momentum (→72 GB, CPU-RAM only) OR fp32 momentum→NVMe | **on** | 72 GB bf16-mom fits CPU RAM; fp32-mom spills 48 GB to NVMe |
| Qwen3.5-4B / 9B (hybrid) | 4/9B | 32/72 GB | T1 | — | on | **blocked**: needs SSM/GDN + gated-attn backward (not ported); adaptive is inference-only here |
| Wan / RxBrain / SenseNova / Krea | — | — | — | — | — | **N/A**: adaptive serving/generation-only, no training recipe |
| TimesFM / TabFM / needle | small | — | R | — | — | **N/A** unless fine-tuning added (no train lane today) |

### Notes per tier decision
- **Resident (≤~1B):** state ≤ 8 GB leaves ~35–40 GB of the 48 GB card for
  activations → train at large batch with no offload, no checkpointing. Fastest
  path; use these to validate the training loop + Muon-at-scale before offload.
- **E4B (T1):** 64 GB state exceeds VRAM but fits CPU RAM. bf16 W (16) + bf16 G
  (16) + fp32 momentum (32) resident in CPU RAM; per-layer W streamed to VRAM
  (double-buffered), grads streamed back, checkpointed activations spill to CPU.
  Micro-batch 1–2 × seq, grad-accum to the target statistical batch. This is the
  "bf16-SGD+ckpt ~32 GB" note made comfortable by moving momentum off-device.
- **gemma-12B (T1+T2):** 96 GB fp32-momentum state exceeds CPU RAM by ~2 GB once
  dataset+activations are counted. Options in preference order: (a) **8-bit/bf16
  momentum** (§2b) → ~24–48 GB momentum, plus **NVFP4/fp8-quantized offloaded
  forward weights** (12B bf16 W 24 GB → ~6 GB NVFP4) → total resident well under
  94 GB CPU RAM, *no NVMe*, and the smaller W makes prefetch compute-bound at
  micro-batch 1; the bf16 master (24 GB) + bf16 grads stay in CPU RAM for the
  update. (b) fallback: keep fp32 momentum, **spill it to NVMe** async-paged during
  the per-group Muon step (touched once/step, ~6 GB/s hidden behind layer
  forward/backward). Prefer (a) — dynamic quant turns the 12B from "T2/NVMe
  required" into "T1 comfortable."

## 4. Implementation dependencies (status → to-do)

The plan runs on primitives that are partly built:

- **Optimizer (Muon/NS) — DONE**, bit-parity vs adaptive (`internal/optimizer`).
- **Quantization codec — DONE** (`internal/quant`, both directions + calibrated +
  in-kernel dequant/fp8-matmul): the dynamic-quant tier (§2b) is enabled today;
  what's missing is wiring it into the *training* offload paths + the quality gate.
- **Host backward VJPs — DONE + FD-verified**: dense (`hostmath`), gemma3n E4B
  (windowed-attn/AltUp/Laurel/PLE, `internal/inference/gemma3n_backward.go`).
- **Resident-model training (≤1B) — VERIFIED**: densecausal grad-parity +
  loss-decrease; diffusion/oscillator/speech step-tests.
- **TO-DO (enabling the offload tiers), in order:**
  1. **Device backward** — overgo's backward is host-only today; the T1/T2 plans
     need GPU backward kernels (the biggest gap).
  2. **Device Newton-Schulz** + per-group streaming — host NS prohibitive ≥500M.
  3. **Layer-streaming offload allocator** — double-buffered CPU↔VRAM weight/grad
     DMA with prefetch, driven by the break-even rule; extends the existing
     device-feed + retained-graph machinery.
  4. **Gradient checkpointing** in the training graph.
  5. **NVMe async pager** for the 12B fp32-momentum fallback + dataset streaming.
  6. **Dynamic-quant training wiring** — apply the existing `internal/quant`
     codec to offloaded forward weights / transferred grads / momentum /
     activation spill (§2b), driven by the break-even rule, gated by measured
     grad-parity + loss-monotonicity (the quality gate).
  7. **Hybrid (qwen3_5) backward** — SSM/GDN + gated-attn VJPs, only if the
     inference-only hybrids are ever to be trained.

Sequencing: land (1)+(2) to move E4B/gemma-12B from host-gradient-verified to
device-trainable at the resident-small scale first, then (3)+(4) for the T1
offload, then (6) dynamic quant to shrink footprint/bandwidth (turns the 12B into
CPU-RAM-comfortable and lowers every model's break-even batch), with (5) NVMe only
as the 12B fp32-momentum fallback. Each step is grad-parity gated against the
FD-verified host backward already in the tree.
