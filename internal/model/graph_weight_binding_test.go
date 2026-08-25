package model

import (
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestCompiledWeightSlotsOwnValidation(t *testing.T) {
	const scope = "compiled graph"

	builder := tensor.NewBuilder()
	weight := builder.Input(scope, dtype.F32, tensor.MustShape(tensor.SingletonExtent))
	weights := graphWeights{weight, weight}
	if err := weights.validate(scope); err != nil {
		t.Fatal(err)
	}
	weights[len(weights)-1] = nil
	if err := weights.validate(scope); err == nil {
		t.Fatal("nil compiled weight slot accepted")
	}
}

func TestBindSequenceOutputGraphWeights(t *testing.T) {
	info := &AudioDecoderWeights{
		InputConv: gguf.TensorInfo{Name: "input"},
		Output:    gguf.TensorInfo{Name: "output"},
		PosNet: []WavPosNetWeights{{
			Norm1:      gguf.TensorInfo{Name: "residual_norm"},
			AttentionQ: gguf.TensorInfo{Name: "residual_query"},
		}},
		ConvNext: []WavConvNextWeights{{
			Depthwise: gguf.TensorInfo{Name: "depthwise"},
			Gamma:     gguf.TensorInfo{Name: "gamma"},
		}},
	}
	builder := tensor.NewBuilder()
	weights, err := BindSequenceOutputGraphWeights(info, func(info gguf.TensorInfo) (*tensor.Tensor, error) {
		return builder.Input(info.Name, dtype.F32, tensor.MustShape(1)), builder.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, node := range map[string]*tensor.Tensor{
		"input": weights.InputConv, "output": weights.Output,
		"residual_norm":  weights.Residual[0].Norm1,
		"residual_query": weights.Residual[0].AttentionQ,
		"depthwise":      weights.Convolution[0].Depthwise,
		"gamma":          weights.Convolution[0].Gamma,
	} {
		if node == nil || node.Name != name {
			t.Fatalf("bound %q = %+v", name, node)
		}
	}
	if weights.Residual[0].Norm1Bias != nil {
		t.Fatal("empty optional tensor was bound")
	}
}
