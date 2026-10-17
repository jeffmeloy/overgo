package tensor

import (
	"testing"

	"overgo/internal/tensor/dtype"
)

// TestNativeTensorCorePolicy checks weight-aware projection policy and routed-expert arithmetic.
func TestNativeTensorCorePolicy(t *testing.T) {
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
		{name: "fp8", weight: dtype.F8E4M3, native: true},
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
	for _, compute := range []MulMatCompute{MulMatComputeExact, MulMatComputeNativeTensorCore} {
		builder := NewBuilder()
		builder.SetMulMatCompute(compute)
		input := builder.Input("input", dtype.F32, shape)
		router := builder.Input("router", dtype.F32, MustShape(32, 4))
		gate := builder.Input("gate", dtype.BF16, MustShape(32, 8, 4))
		up := builder.Input("up", dtype.BF16, gate.Shape)
		down := builder.Input("down", dtype.BF16, MustShape(8, 32, 4))
		output := builder.MoEWithOptions(input, router, up, down, MoEOptions{
			Gate: gate, TopK: 2, Scale: 1, Routing: MoERoutingSoftmax, Activation: MoEActivationSiLU, ExpertIndexDivisor: 1,
		})
		if builder.Err() != nil {
			t.Fatal(builder.Err())
		}
		if got := output.Attrs.(MoEAttributes).Compute; got != compute {
			t.Fatalf("MoE compute = %v, want graph policy %v", got, compute)
		}
	}
}
