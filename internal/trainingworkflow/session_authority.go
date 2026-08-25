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
	// The recipe's objective DOCUMENT is the authority on loss semantics:
	// the module chain names the phase shape, but the store-committed
	// objective names the kind, and a registered flow-matching objective
	// rides the same four-node chain the token bootstrap does. A bound
	// objective dependency FAILS CLOSED -- a recipe that declares an
	// objective it cannot produce is refused, never silently downgraded
	// to module-chain inference.
	if objectiveID, bound := program.Definition().PrimaryDependency(recipe.DependencyObjective); bound {
		document, loadErr := trainingprogram.LoadObjective(ctx, repository, objectiveID)
		if loadErr != nil {
			return SessionAuthority{}, fmt.Errorf("training workflow: recipe objective %s: %w", objectiveID, loadErr)
		}
		objective, err = document.Kind, nil
	}
	if err != nil {
		return SessionAuthority{}, err
	}
	policy, err := trainingprogram.OptimizerPolicyFromRecipe(ctx, repository, program.Definition())
	if err != nil {
		return SessionAuthority{}, err
	}
	return SessionAuthority{Program: program, Objective: objective, Optimizer: policy}, nil
}
