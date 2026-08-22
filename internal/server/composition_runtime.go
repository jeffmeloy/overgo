package server

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/recipe"
)

// OpenProductionComposition resolves one active composition through the
// serving repository. The handler supplies repository authority while the
// caller supplies the live sessions and exact bridge-weight loader.
func (h *Handler) OpenProductionComposition(
	ctx context.Context,
	source, target artifact.ID,
	task recipe.Task,
	resources inference.CompositionRuntimeResources,
) (*inference.ProductionComposition, error) {
	if h == nil || h.repository == nil {
		return nil, errors.New("server: composition repository is absent")
	}
	return inference.OpenProductionComposition(ctx, h.repository, source, target, task, resources)
}

// OpenExternalCrossAttention resolves an active composition and compiles its
// external cache domain against the exact target model plan.
func (h *Handler) OpenExternalCrossAttention(
	ctx context.Context,
	source, target artifact.ID,
	task recipe.Task,
	targetPlan model.ModelPlan,
) (inference.ExternalCrossAttentionProgram, error) {
	if h == nil || h.repository == nil {
		return inference.ExternalCrossAttentionProgram{}, errors.New("server: composition repository is absent")
	}
	return inference.OpenExternalCrossAttention(ctx, h.repository, source, target, task, targetPlan)
}
