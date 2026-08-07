package inference

import (
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestBuildOutputNormUsesCompiledPolicy(t *testing.T) {
	t.Run("absent", func(t *testing.T) {
		profile, _ := model.LookupArchitecture("bert")
		runner := &Runner{preparedModel: preparedModel{
			spec:    model.Spec{CommonSpec: model.CommonSpec{Architecture: "bert"}},
			program: modelrecipe.Plan{Model: model.ModelPlan{Profile: profile}},
		}}
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
		called := false
		output, err := runner.buildOutputNorm(
			builder, input, func(gguf.TensorInfo) (*tensor.Tensor, error) {
				called = true
				return nil, nil
			},
		)
		if err != nil || output != input || called {
			t.Fatalf("output = %p, input = %p, called = %v, error = %v", output, input, called, err)
		}
	})

	t.Run("weighted", func(t *testing.T) {
		profile, _ := model.LookupArchitecture("llama")
		runner := &Runner{preparedModel: preparedModel{
			spec: model.Spec{CommonSpec: model.CommonSpec{
				Architecture: "llama", EmbeddingLength: 2, RMSNormEpsilon: 1e-5,
			}},
			program: modelrecipe.Plan{Model: model.ModelPlan{Profile: profile}},
			weights: model.Weights{OutputNorm: gguf.TensorInfo{Name: "output_norm.weight"}},
		}}
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
		calls := 0
		output, err := runner.buildOutputNorm(
			builder, input, func(info gguf.TensorInfo) (*tensor.Tensor, error) {
				calls++
				return builder.Input(info.Name, dtype.F32, tensor.MustShape(2)), nil
			},
		)
		if err != nil || output == input || calls != 1 || builder.Err() != nil {
			t.Fatalf("output = %p, input = %p, calls = %d, error = %v/%v",
				output, input, calls, err, builder.Err())
		}
	})
}
