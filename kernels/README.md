# CUDA kernels

CUDA source is the explicit non-Go source exception described in
`skill.md`. Runtime and model logic remain Go-owned.

The generated kernel assets must be reproducible. `go run ./cmd/build-kernels`
is the single owner of kernel generation: it compiles every module in its
kernel table to PTX, refreshes the runtime hash pins in
`internal/cuda/kernel/manifest_generated.go`, and regenerates and verifies
`kernels/manifest.json` through `cmd/kernel-manifest`. The tool resolves the
toolchain itself — nvcc from `CUDA_PATH` (falling back to the CUDA 12.9
install location) and a CUDA-compatible MSVC x64 host compiler from the
known Visual Studio Build Tools locations — so no hand-written nvcc
invocation with machine-specific paths is part of the procedure.

CUDA 12.9 rejects the Visual Studio 2026 compiler, so the toolchain probe
prefers the Visual Studio 2019 Build Tools installation. This is a
build-time constraint only; it does not affect the generated PTX.

All modules are built with `-arch=compute_89`. That floor is a deliberate
decision, not a leftover: the only machine whose gated suites execute these
kernels carries a compute-capability-8.9 device, so 8.9 is the highest
target whose behavior the suites actually verify, and PTX built for
compute_89 JIT-compiles forward onto newer devices. Lowering the floor
would publish a compatibility claim for older architectures that no gated
suite measures. The floor moves only when a machine with a different device
runs the CUDA lane and its evidence lands.

`kernels/manifest.json` is the versioned kernel ABI and provenance record. It
pins the schema/ABI version, toolkit and compiler, target, compiler flags,
default launch width, shared-memory convention, source/PTX hashes, and the
complete exported PTX entry set. Its grouped argument layouts record every
entry's ordered PTX parameter types, including explicit alignment and array
size where applicable. Regenerate all assets and hashes with:

```powershell
go run ./cmd/build-kernels
```

Verify checked-in assets without modifying them with:

```powershell
go run ./cmd/kernel-manifest
```

The Go host pins the manifest ABI identity and the PTX hashes in
`internal/cuda/kernel/validation.go`. Module loading validates the embedded
bytes before calling the CUDA driver. After an intentional kernel rebuild,
review the ABI change and update those pins; the manifest verifier will reject
an accidental host/kernel mismatch.
