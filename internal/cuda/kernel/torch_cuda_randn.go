package kernel

import _ "embed"

// TorchCudaRandnPTX: PyTorch-bit-exact standard-normal source (Philox
// torch_cuda_randn). Compiled WITHOUT -use_fast_math so curand's Box-Muller
// matches torch.randn(seed) to the last bit. Loaded by internal/torchrng.
//
//go:embed torch_cuda_randn.ptx
var TorchCudaRandnPTX []byte
