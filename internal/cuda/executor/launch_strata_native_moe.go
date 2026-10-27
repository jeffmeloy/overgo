package executor

import (
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

// Strata's routed-expert combine (kernels/cuda/strata_native_moe.cuh, ported from Strata at 296b59d2, MIT:
// src/kernels/cuda/native_moe.cu), its host side translated.

// native_moe_combine_multi's float4 form.
const (
	strataCombineExperts = 10  // combine_k10_vec4's experts a token
	strataCombineThreads = 128 // its block
)

// launchStrataNativeMoECombineMulti is native_moe_combine_multi at Flash-Next's ten experts: each of tokens rows of
// embedding floats is its ten parts weighted (the first product rounded, then an FMA an expert in order) plus its shared
// row. The float4 form only: the rows must be whole float4s and 16-byte aligned, as Flash-Next's round has them.
func launchStrataNativeMoECombineMulti(state *device.State, functions functionSet, parts, weights, shared, output driver.DevicePtr,
	embedding, experts int64, tokens int) error {
	if experts != strataCombineExperts || shared == 0 || tokens < 1 || embedding%strataCopyFloatsAVector != 0 ||
		(parts|shared|output)%strataCopyVectorAlign != 0 {
		return fmt.Errorf("native MoE combine (multi): only the ten-expert float4 form with a shared row is ported")
	}
	n4 := embedding / strataCopyFloatsAVector
	return launchGridABI(state, functions[kernelStrataNativeCombineK10Vec4],
		driver.Dim3{X: uint32((n4 + strataCombineThreads - 1) / strataCombineThreads), Y: uint32(tokens), Z: 1},
		driver.Dim3{X: strataCombineThreads, Y: 1, Z: 1}, &parts, &weights, &shared, &output, &n4)
}
