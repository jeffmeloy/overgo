package trainingworkflow

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// SessionAuthority binds active recipe topology and optimizer policy.
type SessionAuthority struct {
	Program   recipe.Program
	Objective trainingprogram.ObjectiveKind
	Optimizer trainingprogram.OptimizerPolicy
}

// ResolveSessionAuthority resolves exact active training policy.
func ResolveSessionAuthority(ctx context.Context, repository artifact.Reader, model, recipeID artifact.ID) (SessionAuthority, error) {
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, repository, model, recipe.TaskTraining)
	if err != nil {
		return SessionAuthority{}, fmt.Errorf("training workflow: resolve active recipe: %w", err)
	}
	if program.Definition().ID != recipeID {
		return SessionAuthority{}, errors.New("training workflow: active recipe differs")
	}
	objective, err := ProgramObjective(program)
	if err != nil {
		return SessionAuthority{}, err
	}
	policy, err := trainingprogram.OptimizerPolicyFromRecipe(ctx, repository, program.Definition())
	if err != nil {
		return SessionAuthority{}, err
	}
	return SessionAuthority{Program: program, Objective: objective, Optimizer: policy}, nil
}
