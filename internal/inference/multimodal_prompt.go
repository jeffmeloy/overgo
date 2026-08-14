package inference

import (
	"errors"
	"fmt"

	"overgo/internal/projector"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// ProjectedInputsForPrompt: projector output -> inference admission.
func ProjectedInputsForPrompt(
	runner *Runner,
	prompt projector.MultimodalPrompt,
) ([]tokenizer.TokenID, ProjectedInputs, error) {
	if runner == nil || prompt.EmbeddingWidth != int(runner.Spec().EmbeddingLength) {
		return nil, ProjectedInputs{}, fmt.Errorf(
			"inference: projector width %d differs from model width", prompt.EmbeddingWidth,
		)
	}
	mediaTokens := len(prompt.EmbeddingTokenIndices)
	if len(prompt.Embeddings) != mediaTokens*prompt.EmbeddingWidth {
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
		if len(stream) != mediaTokens*prompt.EmbeddingWidth {
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
		projected.DeepstackEmbeddings[streamIndex] = reference.Value{
			Shape: tensor.MustShape(uint64(prompt.EmbeddingWidth), uint64(len(prompt.TokenIDs))),
			Data:  data,
		}
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
