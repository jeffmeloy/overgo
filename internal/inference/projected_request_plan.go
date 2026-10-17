package inference

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/model"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// Raw scaled media embeddings replace the main input, while their per-layer
// token-identity component comes from the declared padding token. Preserve
// caller prompt IDs and the media embeddings used by the context projection.
func (p projectedRequestPlan) perLayerEmbeddingRows(rows []uint32, padding tokenizer.TokenID) ([]uint32, error) {
	if p.overridePolicy != model.EmbeddingOverrideRawScaled || len(p.overrides) == 0 {
		return rows, nil
	}
	if padding < 0 {
		return nil, errors.New("inference: projected per-layer inputs require a declared padding token")
	}
	selected := slices.Clone(rows)
	for _, override := range p.overrides {
		if uint64(override.TokenIndex) >= uint64(len(selected)) {
			return nil, errors.New("inference: projected per-layer token index is out of range")
		}
		selected[override.TokenIndex] = uint32(padding)
	}
	return selected, nil
}

type projectedRequestPlan struct {
	overridePolicy    model.EmbeddingOverridePolicy
	overrides         []EmbeddingOverride
	multiPositions    *MultiAxisPositions
	deepstackInputs   []reference.Value
	attentionBlockIDs []float32
	visualBlocks      []AttentionBlock
	visualMode        bool
	deepstackBase     bool
}

func (r *Runner) compileProjectedRequestPlan(
	tokens int,
	hasCache bool,
	inputs ProjectedInputs,
) (projectedRequestPlan, error) {
	if tokens == 0 {
		return projectedRequestPlan{}, errors.New("inference: token sequence is empty")
	}
	program := r.program.Model.ProjectedInput()
	if inputs.MultiAxisPositions != nil {
		if !program.MultiAxis {
			return projectedRequestPlan{}, errors.New("inference: model does not support multi-axis positions")
		}
		for axis := range inputs.MultiAxisPositions {
			if len((*inputs.MultiAxisPositions)[axis]) != tokens {
				return projectedRequestPlan{}, fmt.Errorf(
					"inference: multi-axis position %d has %d values for %d tokens",
					axis, len((*inputs.MultiAxisPositions)[axis]), tokens,
				)
			}
		}
	}
	if err := validateDeepstackInputs(r.spec, program.DeepstackStreams, tokens, inputs.DeepstackEmbeddings); err != nil {
		return projectedRequestPlan{}, err
	}
	attentionBlockIDs, err := projectedAttentionBlockIDs(
		program.AttentionBlocks, tokens, hasCache, inputs.BidirectionalAttentionBlocks,
	)
	if err != nil {
		return projectedRequestPlan{}, err
	}
	visualMode := program.Overrides == model.EmbeddingOverrideVisualSpan && len(inputs.EmbeddingOverrides) > 0
	if visualMode {
		if err := validateCogVLMVisualOverrides(tokens, inputs.EmbeddingOverrides); err != nil {
			return projectedRequestPlan{}, err
		}
	}
	var visualBlocks []AttentionBlock
	if len(inputs.VisualExpertBlocks) > 0 {
		if program.Overrides != model.EmbeddingOverrideVisualSpan {
			return projectedRequestPlan{}, errors.New("inference: visual expert blocks require CogVLM architecture")
		}
		if inputs.MultiAxisPositions != nil || len(inputs.DeepstackEmbeddings) > 0 ||
			len(inputs.BidirectionalAttentionBlocks) > 0 {
			return projectedRequestPlan{}, errors.New("inference: CogVLM visual expert blocks cannot combine with MRoPE, deepstack, or bidirectional blocks")
		}
		visualBlocks, err = validateVisualExpertBlocks(tokens, inputs.VisualExpertBlocks, inputs.EmbeddingOverrides)
		if err != nil {
			return projectedRequestPlan{}, err
		}
	}
	return projectedRequestPlan{
		overridePolicy: program.Overrides, overrides: inputs.EmbeddingOverrides,
		multiPositions: inputs.MultiAxisPositions, deepstackInputs: inputs.DeepstackEmbeddings,
		attentionBlockIDs: attentionBlockIDs, visualBlocks: visualBlocks,
		visualMode:    visualMode,
		deepstackBase: program.Overrides == model.EmbeddingOverrideMappedBase && len(r.spec.DeepstackMapping) > 0,
	}, nil
}
