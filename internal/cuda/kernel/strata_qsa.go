package kernel

import _ "embed"

// StrataQsaPTX contains Strata's QSA attention layer kernels, ported literally (kernels/cuda/strata_qsa.cu, MIT,
// licenses/Strata-MIT.txt). Compiled WITHOUT -use_fast_math, as Strata compiles them, so the quantizer's division
// and the attention's arithmetic keep the bits Strata's parity checks hold. Loaded by internal/cuda/executor
// beside the operation module.
//
//go:embed strata_qsa.ptx
var StrataQsaPTX []byte
