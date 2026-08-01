package model

import (
	"math"
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

func TestBuildQwen35MTPPipeline(t *testing.T) {
	for _, architecture := range []string{"qwen35", "qwen35moe"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := qwen35TestSpec()
			spec.Architecture = architecture
			spec.NextNPredictLayers = 1
			token := builder.Input("token", dtype.F32, tensor.MustShape(8, 1))
			hidden := builder.Input("hidden", dtype.F32, tensor.MustShape(8, 1))
			embeddingNorm := builder.Input("enorm", dtype.F32, tensor.MustShape(8))
			hiddenNorm := builder.Input("hnorm", dtype.F32, tensor.MustShape(8))
			projection := builder.Input("eh", dtype.F32, tensor.MustShape(16, 8))
			current, err := BuildQwen35MTPInput(
				builder, token, hidden, embeddingNorm, hiddenNorm, projection, spec,
			)
			if err != nil {
				t.Fatal(err)
			}
			weightSpec := spec
			weightSpec.Architecture = "qwen35"
			block, err := BuildQwen35MTPBlockCached(
				builder, current, spec, qwen35AttentionInputs(builder, weightSpec), []uint32{7}, nil, nil,
			)
			if err != nil {
				t.Fatal(err)
			}
			outputNorm := builder.Input("output_norm", dtype.F32, tensor.MustShape(8))
			head := builder.Input("head", dtype.F32, tensor.MustShape(8, 13))
			logits, nextHidden, err := BuildQwen35MTPOutputs(builder, block.Output, outputNorm, head, spec)
			if err != nil {
				t.Fatal(err)
			}
			outputs := []*tensor.Tensor{logits, nextHidden, block.Key, block.Value}
			nodes, err := tensor.Topological(outputs...)
			if err != nil {
				t.Fatal(err)
			}
			feeds := make(map[*tensor.Tensor]reference.Value)
			for _, node := range nodes {
				if node.Op != tensor.OpInput {
					continue
				}
				elements, elementErr := node.Shape.Elements()
				if elementErr != nil {
					t.Fatal(elementErr)
				}
				data := make([]float32, elements)
				for index := range data {
					data[index] = 0.05 + float32(index%11)*0.01
				}
				feeds[node] = reference.Value{Shape: node.Shape, Data: data}
			}
			results, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			if !results[logits].Shape.Equal(tensor.MustShape(13, 1)) ||
				!results[nextHidden].Shape.Equal(tensor.MustShape(8, 1)) ||
				!results[block.Key].Shape.Equal(tensor.MustShape(4, 1, 1)) {
				t.Fatalf("unexpected Qwen3.5 MTP shapes: %v %v %v", results[logits].Shape, results[nextHidden].Shape, results[block.Key].Shape)
			}
			for _, value := range results[logits].Data {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("non-finite Qwen3.5 MTP logit: %v", value)
				}
			}
		})
	}
}
