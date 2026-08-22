package inference

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/model"
	"overgo/internal/tensor/reference"
)

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
	if !checked.Nonzero(tokens) {
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
	visualMode := program.Overrides == model.EmbeddingOverrideVisualSpan && checked.Nonzero(len(inputs.EmbeddingOverrides))
	if visualMode {
		if err := validateCogVLMVisualOverrides(tokens, inputs.EmbeddingOverrides); err != nil {
			return projectedRequestPlan{}, err
		}
	}
	var visualBlocks []AttentionBlock
	if checked.Nonzero(len(inputs.VisualExpertBlocks)) {
		if program.Overrides != model.EmbeddingOverrideVisualSpan {
			return projectedRequestPlan{}, errors.New("inference: visual expert blocks require CogVLM architecture")
		}
		if inputs.MultiAxisPositions != nil || checked.Nonzero(len(inputs.DeepstackEmbeddings)) ||
			checked.Nonzero(len(inputs.BidirectionalAttentionBlocks)) {
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
		deepstackBase: program.Overrides == model.EmbeddingOverrideMappedBase && checked.Nonzero(len(r.spec.DeepstackMapping)),
	}, nil
}
