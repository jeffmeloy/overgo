package tensor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

// The native tensor-core policy binds half-precision weights against F32
// activations, which cuBLAS multiplies in their dtype, and quantized
// weights, which the CUDA executor stages to F32 and multiplies exactly
// past its column floor; an F32 or fp8 mul_mat in the same graph keeps
// its exact arithmetic unmarked, so a graph-wide default is safe for a
// transformer forward.
func TestNativeTensorCorePolicyAppliesOnlyToHalfPrecisionWeights(t *testing.T) {
	builder := NewBuilder()
	builder.SetMulMatCompute(MulMatComputeNativeTensorCore)
	shape := MustShape(32, 4)
	activation := builder.Input("activation", dtype.F32, shape)
	cases := []struct {
		name   string
		weight dtype.Type
		native bool
	}{
		{name: "f16", weight: dtype.F16, native: true},
		{name: "bf16", weight: dtype.BF16, native: true},
		{name: "f32", weight: dtype.F32, native: false},
		{name: "q8_0", weight: dtype.Q8_0, native: true},
		{name: "fp8", weight: dtype.F8E4M3, native: false},
	}
	for _, tc := range cases {
		weight := builder.Input(tc.name, tc.weight, shape)
		output := builder.MulMat(weight, activation)
		if builder.Err() != nil {
			t.Fatalf("%s: %v", tc.name, builder.Err())
		}
		attributes, bound := output.Attrs.(MulMatAttributes)
		if bound != tc.native || bound && attributes.Compute != MulMatComputeNativeTensorCore {
			t.Errorf("%s: attributes = %+v, want native=%v", tc.name, output.Attrs, tc.native)
		}
	}
	explicit := NewBuilder()
	weight := explicit.Input("weight", dtype.F32, shape)
	explicit.MulMatWithCompute(weight, activation, MulMatComputeNativeTensorCore)
	if explicit.Err() != nil {
		t.Fatalf("explicit native policy over F32 refused instead of staying exact: %v", explicit.Err())
	}
}
