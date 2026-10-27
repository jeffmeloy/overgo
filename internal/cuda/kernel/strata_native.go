package kernel

import _ "embed"

// StrataNativePTX contains the native kernels of Strata's decode round, ported literally (kernels/cuda/strata_native.cu,
// MIT, licenses/Strata-MIT.txt). Compiled WITH -use_fast_math and as C++17, as Strata compiles its native_* files.
// Loaded by internal/cuda/executor beside the operation module and Strata's QSA kernels.
//
//go:embed strata_native.ptx
var StrataNativePTX []byte
