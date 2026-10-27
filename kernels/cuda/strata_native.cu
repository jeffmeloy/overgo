// strata_native.cu - the module of Strata's native kernels its decode round launches (strata-port/decode).
//
// Ported from Strata (https://github.com/Niko1221/Strata, commit 296b59d2), MIT License, Copyright (c) 2026 Niko1221
// and the Strata contributors (licenses/Strata-MIT.txt); each included file names its sources.
//
// Strata compiles its native_* files WITH --use_fast_math and as C++17 (their templates take if constexpr), so this
// module is compiled the same way, apart from the operation module, whose kernels keep the dialect they were
// compiled in.

#include <cfloat>
#include <cuda_fp16.h>

#include "strata_native_mmvq.cuh"
#include "strata_native_bf16.cuh"
#include "strata_native_gr_postops.cuh"
#include "strata_native_qsa.cuh"
#include "strata_native_router.cuh"
#include "strata_native_ple.cuh"
#include "strata_native_moe.cuh"
