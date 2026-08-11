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

## Non-goals for rung 2
Multi-GPU (single 4090D). Quantized training transport (later rung, after the fp32
device baseline). Changing the host oracle's fp64 determinism.
