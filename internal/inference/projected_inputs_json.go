package inference

import (
	"fmt"
	"math"
	"slices"

	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// ProjectedInputsJSON: portable projected-prompt payload.
type ProjectedInputsJSON struct {
	EmbeddingOverrides           []ProjectedEmbeddingOverrideJSON `json:"embedding_overrides"`
	MultiAxisPositions           *MultiAxisPositions              `json:"multi_axis_positions"`
	DeepstackEmbeddings          []ProjectedTensorJSON            `json:"deepstack_embeddings"`
	BidirectionalAttentionBlocks []AttentionBlock                 `json:"bidirectional_attention_blocks"`
	VisualExpertBlocks           []AttentionBlock                 `json:"visual_expert_blocks"`
}

type ProjectedEmbeddingOverrideJSON struct {
	TokenIndex uint32    `json:"token_index"`
	Embedding  []float32 `json:"embedding"`
}

type ProjectedTensorJSON struct {
	Shape []uint64  `json:"shape"`
	Data  []float32 `json:"data"`
}

// ProjectedInputs: validated runtime payload.
func (document ProjectedInputsJSON) ProjectedInputs() (ProjectedInputs, error) {
	result := ProjectedInputs{
		MultiAxisPositions: document.MultiAxisPositions,
		BidirectionalAttentionBlocks: append(
			[]AttentionBlock(nil), document.BidirectionalAttentionBlocks...,
		),
		VisualExpertBlocks: slices.Clone(document.VisualExpertBlocks),
	}
	result.EmbeddingOverrides = make([]EmbeddingOverride, len(document.EmbeddingOverrides))
	for index, item := range document.EmbeddingOverrides {
		if len(item.Embedding) == 0 || !finiteProjectedFloats(item.Embedding) {
			return ProjectedInputs{}, fmt.Errorf("projected embedding override %d is empty or non-finite", index)
		}
		result.EmbeddingOverrides[index] = EmbeddingOverride{
			TokenIndex: item.TokenIndex,
			Embedding:  slices.Clone(item.Embedding),
		}
	}
	result.DeepstackEmbeddings = make([]reference.Value, len(document.DeepstackEmbeddings))
	for index, item := range document.DeepstackEmbeddings {
		shape, err := tensor.NewShape(item.Shape...)
		if err != nil {
			return ProjectedInputs{}, fmt.Errorf("projected deepstack tensor %d shape: %w", index, err)
		}
		if !finiteProjectedFloats(item.Data) {
			return ProjectedInputs{}, fmt.Errorf("projected deepstack tensor %d contains non-finite data", index)
		}
		value, err := reference.NewValue(shape, item.Data)
		if err != nil {
			return ProjectedInputs{}, fmt.Errorf("projected deepstack tensor %d: %w", index, err)
		}
		result.DeepstackEmbeddings[index] = value
	}
	return result, nil
}

func finiteProjectedFloats(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}
