package trainingworkflow

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// SessionAuthority binds recipe topology and optimizer policy.
type SessionAuthority struct {
	Program   recipe.Program
	Objective trainingprogram.ObjectiveKind
	Optimizer trainingprogram.OptimizerPolicy
}

// trainableStatuses are the lifecycle states a training run may train a
// recipe in. A run's own success is what verifies and activates a candidate.
// Superseded is among them for the recipe that lost its model's one active
// training slot to another recipe's activation: a model keeps one active
// training recipe, a run under another objective's default activates its own,
// and the next run of the first objective trains under the recipe that lost
// the slot. A recipe retired on failed proof is superseded too, and is told
// from that one by the refusal its own record cites.
var trainableStatuses = []recipe.Status{recipe.StatusCandidate, recipe.StatusValidated, recipe.StatusActive, recipe.StatusSuperseded}

// ResolveSessionAuthority resolves the exact training policy of the
// published training recipe recipeID for model, when the recipe's own record
// leaves it eligible: in a state a run may train in, and refused by no
// decision. Which recipe the model has selected plays no part, so selecting
// another recipe neither admits a retired one nor refuses one in good
// standing.
func ResolveSessionAuthority(ctx context.Context, repository artifact.Reader, model, recipeID artifact.ID) (SessionAuthority, error) {
	eligibility, err := modelrecipe.RecipeEligibility(ctx, repository, recipeID)
	if err != nil {
		return SessionAuthority{}, err
	}
	if eligibility.Refusal != "" {
		return SessionAuthority{}, fmt.Errorf("training workflow: recipe %s was refused and no run may train under it until it is activated again on fresh proof: %s", recipeID, eligibility.Refusal)
	}
	if !eligibility.Published || !slices.Contains(trainableStatuses, eligibility.Status) {
		return SessionAuthority{}, fmt.Errorf("training workflow: recipe %s is %q, not a published training recipe a run may train under", recipeID, eligibility.Status)
	}
	definition, err := recipe.RequireDefinition(ctx, repository, recipeID)
	if err != nil {
		return SessionAuthority{}, err
	}
	if definition.Model != model || definition.Task != recipe.TaskTraining {
		return SessionAuthority{}, errors.New("training workflow: recipe model or task differs")
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		return SessionAuthority{}, err
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
