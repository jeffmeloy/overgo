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
// recipe in: published and not retired. A run's own success is what
// verifies and activates a candidate.
var trainableStatuses = []recipe.Status{recipe.StatusCandidate, recipe.StatusValidated, recipe.StatusActive}

// supersededBySibling reports whether recipeID, in status, lost the model's
// active training slot to another recipe that is active and verified now, not to
// a retirement with failed proof, which leaves the slot naming the retired one.
func supersededBySibling(ctx context.Context, repository artifact.Reader, model, recipeID artifact.ID, status recipe.Status) bool {
	if status != recipe.StatusSuperseded {
		return false
	}
	active, found, err := modelrecipe.ActiveRecord(ctx, repository, model, recipe.TaskTraining)
	return err == nil && found && active.Definition.ID != recipeID
}

// ResolveSessionAuthority resolves the exact training policy of the
// published, unretired training recipe recipeID for model. A recipe that
// another training recipe of the model superseded by activating is not
// retired: a model keeps one active training recipe, a run under another
// objective's default activates its own, and the next run of the first
// objective trains under the recipe it lost the slot to a live successor.
func ResolveSessionAuthority(ctx context.Context, repository artifact.Reader, model, recipeID artifact.ID) (SessionAuthority, error) {
	status, published, err := modelrecipe.Status(ctx, repository, recipeID)
	if err != nil {
		return SessionAuthority{}, err
	}
	if !published || !slices.Contains(trainableStatuses, status) && !supersededBySibling(ctx, repository, model, recipeID, status) {
		return SessionAuthority{}, fmt.Errorf("training workflow: recipe %s is %q, not a published training recipe a run may train under", recipeID, status)
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
