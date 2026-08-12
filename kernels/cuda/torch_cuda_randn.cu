// torch_cuda_randn: PyTorch-bit-exact standard normal source. Ported verbatim
// from adaptive_new go/extmodel (NewTorchCUDARandnCUDAStream). The Philox
// counter-based generator plus curand's float4 Box-Muller fill-order is what
// makes the output byte-match torch.randn(seed) on CUDA -- do NOT "improve" the
// algorithm or reorder the fill.
//
// ABI note: this module MUST be compiled WITHOUT -use_fast_math. curand_normal4
// draws normals through logf/sqrtf/sincospif; the fast-math intrinsics change
// those last bits and break parity with PyTorch (which links standard-precision
// curand). cmd/build-kernels compiles this .cu with fast-math disabled.
#include <curand_kernel.h>

extern "C" __global__ void torch_cuda_randn(float* out, long long numel, unsigned long long seed, unsigned long long offset) {
    const int unroll = 4;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    long long stride = (long long)blockDim.x * gridDim.x * unroll;
    long long rounded = ((numel - 1) / stride + 1) * stride;
    for (long long linear = idx; linear < rounded; linear += stride) {
        float4 r = curand_normal4(&state);
        long long li = linear;
        if (li < numel) out[li] = r.x;
        li += (long long)blockDim.x * gridDim.x;
        if (li < numel) out[li] = r.y;
        li += (long long)blockDim.x * gridDim.x;
        if (li < numel) out[li] = r.z;
        li += (long long)blockDim.x * gridDim.x;
        if (li < numel) out[li] = r.w;
    }
}
