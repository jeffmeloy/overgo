# CUDA kernels

CUDA source is the explicit non-Go source exception described in
`skill.md`. Runtime and model logic remain Go-owned.

The generated kernel assets must be reproducible. The initial smoke-test PTX is
built for compute capability 8.9:

```powershell
nvcc -ptx -arch=compute_89 `
  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" `
  -o internal\cuda\kernel\vector_add.ptx `
  kernels\cuda\vector_add.cu
```

CUDA 12.9 rejects the installed Visual Studio 2026 compiler. The compatible
Visual Studio 2019 Build Tools installation is therefore used for kernel
generation. This is a build-time constraint only.

`kernels/manifest.json` is the versioned kernel ABI and provenance record. It
pins the schema/ABI version, toolkit and compiler, target, compiler flags,
default launch width, shared-memory convention, source/PTX hashes, and the
complete exported PTX entry set. Its grouped argument layouts record every
entry's ordered PTX parameter types, including explicit alignment and array
size where applicable. Regenerate all assets and hashes with:

```powershell
.\scripts\build-kernels.ps1
```

Verify checked-in assets without modifying them with:

```powershell
go run ./cmd/kernel-manifest
```

The Go host pins the manifest ABI identity and both PTX hashes in
`internal/cuda/kernel/validation.go`. Module loading validates the embedded
bytes before calling the CUDA driver. After an intentional kernel rebuild,
review the ABI change and update those pins; the manifest verifier will reject
an accidental host/kernel mismatch.
