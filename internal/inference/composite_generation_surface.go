package inference

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/recipe"
)

// CompositeGenerationSurface is the immutable selection returned to runtime,
// API, and GUI. It contains no caller-selected recipe or direct constructor.
type CompositeGenerationSurface struct {
	Source    artifact.ID `json:"source"`
	Target    artifact.ID `json:"target"`
	Task      recipe.Task `json:"task"`
	Recipe    artifact.ID `json:"recipe"`
	Plan      artifact.ID `json:"plan"`
	Promotion artifact.ID `json:"promotion"`
	Evidence  artifact.ID `json:"evidence"`
	Output    artifact.ID `json:"output"`
}

// ResolveCompositeGenerationSurface resolves the sole promoted output through
// the current source-target-task active aliases. API and GUI use this same
// runtime resolver, so neither surface can substitute a recipe or artifact.
func ResolveCompositeGenerationSurface(
	ctx context.Context,
	reader artifact.Reader,
	source, target artifact.ID,
	task recipe.Task,
) (CompositeGenerationSurface, error) {
	if ctx == nil || reader == nil {
		return CompositeGenerationSurface{}, errors.New("inference: composite generation surface authority is absent")
	}
	promotion, plan, found, err := composition.ActiveCompositeGeneration(ctx, reader, source, target, task)
	if err != nil {
		return CompositeGenerationSurface{}, err
	}
	if !found {
		return CompositeGenerationSurface{}, errors.New("inference: composite generation is not evidence-promoted")
	}
	return compositeGenerationSurface(plan, promotion), nil
}

// PromotedGeneration resolves the surface selection against the exact plan
// already retained by a live production composition.
func (runtime *ProductionComposition) PromotedGeneration(
	ctx context.Context,
	reader artifact.Reader,
) (CompositeGenerationSurface, error) {
	if runtime == nil || runtime.plan.ValidateIdentity() != nil {
		return CompositeGenerationSurface{}, errors.New("inference: production composition is absent")
	}
	promotion, plan, found, err := composition.ActiveCompositeGeneration(
		ctx, reader, runtime.plan.SourceModel, runtime.plan.TargetModel, runtime.plan.Task,
	)
	if err != nil {
		return CompositeGenerationSurface{}, err
	}
	if !found {
		return CompositeGenerationSurface{}, errors.New("inference: production composition is not evidence-promoted")
	}
	if plan.ID != runtime.plan.ID || plan.CompositionRecipe != runtime.plan.CompositionRecipe {
		return CompositeGenerationSurface{}, errors.New("inference: promoted generation differs from resident composition")
	}
	return compositeGenerationSurface(plan, promotion), nil
}

func compositeGenerationSurface(
	plan composition.CompositionExecutionPlan,
	promotion composition.CompositeGenerationPromotion,
) CompositeGenerationSurface {
	return CompositeGenerationSurface{
		Source: plan.SourceModel, Target: plan.TargetModel, Task: plan.Task,
		Recipe: plan.CompositionRecipe, Plan: plan.ID, Promotion: promotion.ID,
		Evidence: promotion.CUDAEvidence, Output: promotion.Output,
	}
}
