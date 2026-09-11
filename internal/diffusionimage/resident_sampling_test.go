package diffusionimage

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestResidentSamplingContractAcceptance(t *testing.T) {
	// The scalar is broadcast without changing the reference's two rounded
	// operations. An explicit F32 conversion prevents a host fused multiply-add.
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(3, 2)
	input := builder.Input("input", dtype.F32, shape)
	velocity := builder.Input("velocity", dtype.F32, shape)
	delta, next := samplingUpdate(builder, input, velocity)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	x := []float32{1, -2, 0, 7, -8, 0.125}
	v := []float32{-3, 2, 9, -0.125, 4, 11}
	for _, steps := range []int{1, 3, 8} {
		dt := float32(1) / float32(steps)
		result, err := reference.Execute([]*tensor.Tensor{next}, map[*tensor.Tensor]reference.Value{
			input: {Shape: shape, Data: x}, velocity: {Shape: shape, Data: v},
			delta: {Shape: delta.Shape, Data: []float32{dt}},
		})
		if err != nil {
			t.Fatal(err)
		}
		for index, value := range result[next].Data {
			if want := x[index] + float32(v[index]*dt); value != want {
				t.Fatalf("steps=%d element=%d got=%g want=%g", steps, index, value, want)
			}
		}
	}
	var unavailable *ResidentForward
	if _, err := unavailable.Sample(t.Context(), 1, 0); err == nil {
		t.Fatal("unavailable sampler accepted")
	}
	if err := unavailable.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}
