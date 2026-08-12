package model

import (
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestBuildQwen35MTPPipeline(t *testing.T) {
	for _, architecture := range []string{"qwen35", "qwen35moe"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := qwen35TestSpec()
			spec.Architecture = architecture
			spec.NextNPredictLayers = 1
			draft := fixtureDraftProgram(t, spec, Weights{}, 0)
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
			plan := draft.Layer()
			block, err := draft.Build(CachedBlockContext{
				Builder: builder, Input: current, Positions: []uint32{7}, Layer: plan.Layer,
			}, qwen35AttentionInputs(builder, weightSpec))
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
