package inference

import (
	"errors"
	"fmt"

	"overgo/internal/projector"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// CompileProjectedInputs: projector output -> inference admission.
func CompileProjectedInputs(
	prompt projector.MultimodalPrompt,
	embeddingWidth uint32,
) ([]tokenizer.TokenID, ProjectedInputs, error) {
	mediaTokens := len(prompt.EmbeddingTokenIndices)
	shape, shapeErr := tensor.NewShape(uint64(prompt.EmbeddingWidth), uint64(mediaTokens))
	embeddings := reference.Value{Shape: shape, Data: prompt.Embeddings}
	if shapeErr != nil || !embeddings.IsMatrixWidth(uint64(embeddingWidth)) {
		return nil, ProjectedInputs{}, fmt.Errorf(
			"inference: projector width %d differs from model width", prompt.EmbeddingWidth,
		)
	}
	if validateStateValue(embeddings) != nil {
		return nil, ProjectedInputs{}, errors.New("inference: projector embedding indices are invalid")
	}
	overrides := make([]EmbeddingOverride, mediaTokens)
	for index := range overrides {
		start := index * prompt.EmbeddingWidth
		overrides[index] = EmbeddingOverride{
			TokenIndex: prompt.EmbeddingTokenIndices[index],
			Embedding:  prompt.Embeddings[start : start+prompt.EmbeddingWidth],
		}
	}
	projected := ProjectedInputs{EmbeddingOverrides: overrides}
	projected.DeepstackEmbeddings = make([]reference.Value, len(prompt.DeepstackEmbeddings))
	for streamIndex, stream := range prompt.DeepstackEmbeddings {
		if validateStateValue(reference.Value{Shape: shape, Data: stream}) != nil {
			return nil, ProjectedInputs{}, fmt.Errorf("inference: projector deepstack stream %d is invalid", streamIndex)
		}
		data := make([]float32, len(prompt.TokenIDs)*prompt.EmbeddingWidth)
		for index, tokenIndex := range prompt.EmbeddingTokenIndices {
			if uint64(tokenIndex) >= uint64(len(prompt.TokenIDs)) {
				return nil, ProjectedInputs{}, errors.New("inference: projector embedding index exceeds prompt")
			}
			source := stream[index*prompt.EmbeddingWidth : (index+1)*prompt.EmbeddingWidth]
			copy(data[int(tokenIndex)*prompt.EmbeddingWidth:], source)
		}
		fullShape, err := tensor.NewShape(uint64(embeddingWidth), uint64(len(prompt.TokenIDs)))
		if err != nil {
			return nil, ProjectedInputs{}, errors.New("inference: projector prompt shape is invalid")
		}
		projected.DeepstackEmbeddings[streamIndex] = reference.Value{Shape: fullShape, Data: data}
	}
	projected.BidirectionalAttentionBlocks = make([]AttentionBlock, len(prompt.AttentionBlocks))
	for index, block := range prompt.AttentionBlocks {
		projected.BidirectionalAttentionBlocks[index] = AttentionBlock{Start: block.Start, End: block.End}
	}
	projected.VisualExpertBlocks = make([]AttentionBlock, len(prompt.VisualBlocks))
	for index, block := range prompt.VisualBlocks {
		projected.VisualExpertBlocks[index] = AttentionBlock{Start: block.Start, End: block.End}
	}
	hasMultiAxis := false
	for _, axis := range prompt.MultiAxisPositions {
		hasMultiAxis = hasMultiAxis || len(axis) > 0
	}
	if hasMultiAxis {
		positions := MultiAxisPositions(prompt.MultiAxisPositions)
		projected.MultiAxisPositions = &positions
	}
	return prompt.TokenIDs, projected, nil
}
