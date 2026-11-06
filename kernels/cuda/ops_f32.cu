// ops_f32.cu is the F32 kernel module, built with its own compiler flags. Its kernels lie in family headers it
// includes in their order, so the module compiles as one text.
extern "C" __device__ float __nv_exp2f(float);
extern "C" __device__ float __nv_expf(float);
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include "attention_row.cuh"
#include "bf16_pack.cuh"
#include "iq_tables_generated.cuh"
#include "strata_qsa_native.cuh"

#include "ops_f32_common.cuh"
#include "ops_f32_elementwise.cuh"
#include "ops_f32_activations.cuh"
#include "ops_f32_vision.cuh"
#include "ops_f32_unary.cuh"
#include "ops_f32_recurrent.cuh"
#include "ops_f32_selection.cuh"
#include "ops_f32_norms.cuh"
#include "ops_f32_rows.cuh"
#include "ops_f32_routing.cuh"
#include "ops_f32_attention.cuh"
#include "ops_f32_quant.cuh"
#include "ops_f32_experts.cuh"
#include "ops_f32_vae.cuh"
#include "ops_f32_tail.cuh"
