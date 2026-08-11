# Rung 2 — device (CUDA) Muon training

Goal: move the training compute onto the GPU so it matches adaptive_new's speed
class and beats it on memory, while keeping the Go build cgo-free. The host loop
already shipped (densecausal.Train, derived LR, exact checkpoint/resume) is the
**correctness oracle**: every device slice is gated against it.

## Approach (per user direction)
- **Baseline = adaptive_new's Go CUDA training** (NVRTC-compiled kernels in
  go/extmodel/*_cuda_windows.go): same algorithm, coefficients, and geometry.
- **Beat it with overgo's existing optimizations**: the cgo-free cuBLAS wrapper
  (internal/cuda/cublas.GEMMEx), the committed-PTX launch path
  (nvcuda.dll via internal/cuda/driver, no cgo), native-dtype compute, and
  llama.cpp-derived kernels already in internal/cuda. Build new kernels only for
  ops none of those cover.

## Grounded infrastructure (verified present)
- GPU compute is cgo-free: `internal/cuda/driver` loads `nvcuda.dll`
  (`syscall.LoadDLL` + `cuModuleLoadData`/`cuLaunchKernel`); memory via
  `MemAlloc`/`MemcpyHtoD`/`MemcpyDtoH`/`MemFree`.
- `internal/cuda/cublas.GEMMEx` (column-major mixed-storage GEMM) + `Sgemm` /
  `SgemmStridedBatched`.
- `internal/cuda/device.Worker.Do(ctx, func(*State) error)` pins a device context
  on a goroutine; `internal/cuda/executor` is the inference precedent that already
  composes cuBLAS + custom kernels + device memory.
- New kernels: author `.cu` -> offline `nvcc -ptx` (CUDA 12.9 present) -> commit
  `.ptx` -> runtime launch. Same model as `kernels/cuda/ops_f32.cu`.

## Precision / parity contract
Host Newton-Schulz + optimizer run in **fp64** (the oracle). Device runs in
**fp32** (the baseline's training precision; §3.2 of the training plan). Parity is
therefore **tolerance-gated**, not bitwise: report max/rel error of the device
result against the fp64 host result, with the tolerance recorded per slice.
Bitwise host determinism (checkpoint/resume) is unchanged.

## Slice decomposition (each host-gated; ranked lowest-risk / highest-unblock first)

1. **Device Newton-Schulz via cuBLAS** *(first slice)*. NS is normalize + 10
   iterations of {gram = XᵀX, square = gram², polynomial c0·I+c1·gram+c2·square,
   out = X·square} — all GEMM + one elementwise. Map the three matmuls to
   `cublas.GEMMEx` (fp32; handle row-major via the Cᵀ=BᵀAᵀ transpose convention),
   and the normalize + polynomial to one small committed-PTX elementwise kernel.
   Gate: device NS vs host `newtonSchulz` on random matrices across shapes
   (tall/wide/square, sizes to 4096), rel-error tolerance. **Directly removes the
   host-NS-on-500M timeout wall.**
2. **Device Muon step.** Compose slice 1 with nesterov momentum + per-group RMS
   scaling into a device `optimizer.Step`; gate vs host `optimizer.Step` on a
   multi-group plan.
3. **Device backward, per operator** (the long pole; decompose by op: matmul,
   RMSNorm, SwiGLU, attention). Each kernel grad-parity gated against the
   FD-verified host VJP in `internal/hostmath`. Reuse inference forward kernels
   from `internal/cuda/executor` where the shapes match.
4. **Device forward + loss** for training (reuse the inference executor's forward;
   add the causal-LM loss + dLogits on device).
5. **End-to-end device training loop.** Gate the full device trajectory against
   host `densecausal.Train` within tolerance; verify checkpoint/resume; then run
   the python/go(adaptive_new)/overgo **performance** comparison (speed + peak
   memory) — the milestone the whole rung exists to reach.

## Progress

Slices 1-2 DONE and gate/GPU-verified: device Newton-Schulz (internal/optimizer,
fp32 vs fp64 host ~1e-7), device Muon step (matrix + full-plan `DeviceMuonStepPlan`
vs host `Optimizer.Step`), and it is wired into `densecausal.TrainDevice`
(reproduces the host learning curve to 3.8e-6). Slice 3 (device backward) DONE as
an FD-verified operator library in `internal/devicemath`, each grad-checked on the
GPU against a float64 reference: `LinearBackward`, `SiLUBackward`,
`GatedMLPBackward` (SwiGLU), `RMSNormBackward`, `SoftmaxBackward`,
`AttentionCoreBackward` + `MultiHeadAttentionBackward` (causal GQA),
`RoPEHalfBackward`, and the composed `LayerBackward` (a full generic pre-norm
transformer layer). The ops_f32 kernel-extension path is established (silu/rms-norm/
softmax/rope backward kernels added via `cmd/build-kernels`).

## Remaining: densecausal-exact integration (the next slices)

`devicemath.LayerBackward` proves the operators compose, but densecausal's layer
(internal/densecausal/forward.go attnSubForward/layerForward) uses specific
conventions the integrated backward MUST match to gate against host `layerBackward`:
- **Linear layout is `W[out,in]`, `Y = X·Wᵀ`** (hostmath.Linear), the transpose of
  devicemath.LinearBackward's `Y=X·W` — so its dX/dW GEMMs flip (dX=dY·W,
  dW=dYᵀ·X). Add a `LinearBackwardT` or transpose W at the boundary.
- **Scale is folded into `qScaled`**: q is projected, bias added, RoPE'd, then
  multiplied by `1/sqrt(headDim)`; CausalAttention then runs at scale=1. So the
  device attention backward uses scale=1 and RoPEHalfBackward + the scale factor
  apply to dQ.
- Bias (qb/kb/vb) lands BEFORE rope; RoPE is split-half per head on qPost/kPost.
- attnCore is `[seq, Heads*HeadDim]`; the o-projection maps it to `[seq, Hidden]`.
- densecausal.layerBackward RECOMPUTES attnSubForward for intermediates (checkpoint
  posture) rather than saving them; the device path can mirror that or save a cache.

Then: end-to-end device training loop (device forward + device backward + device
Muon step) gated vs host `densecausal.Train`; then the python/go/overgo speed+memory
comparison — the milestone the whole rung exists to reach.

## Non-goals for rung 2
Multi-GPU (single 4090D). Quantized training transport (later rung, after the fp32
device baseline). Changing the host oracle's fp64 determinism.
