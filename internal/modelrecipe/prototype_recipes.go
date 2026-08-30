package modelrecipe

import (
	"errors"
	"fmt"

	"overgo/internal/recipe"
)

// DerivePrototypeRecipes derives the candidate recipe set for one
// prototype-produced model from its resolved definition and the compiled
// module catalogs. The inference candidate binds the exact model, profile,
// and definition identities with the requested placement, session, and
// residency; every additional task candidate derives through the registered
// capability catalog. Recipes stay immutable recipe definitions — closed
// node and edge graphs over compiled Go modules — and a task the catalog
// does not register refuses instead of selecting a fallback topology;
// nothing embeds code or discovers filesystem instructions.
func DerivePrototypeRecipes(
	resolved ResolvedModelDefinition,
	placement recipe.Placement,
	session DecodeSessionPolicy,
	residency recipe.ResidencyPolicy,
	tasks []recipe.Task,
) (map[recipe.Task]recipe.Definition, error) {
	if !resolved.Document.ID.Valid() || !resolved.Document.Model.Valid() {
		return nil, errors.New("model recipe: recipe derivation requires a resolved model definition")
	}
	inferenceDefinition, err := InferenceWithModelDefinition(
		resolved.Document.Model, resolved.Profile.ID, resolved.Document.ID,
		placement, session, residency,
	)
	if err != nil {
		return nil, err
	}
	candidates := map[recipe.Task]recipe.Definition{recipe.TaskInference: inferenceDefinition}
	for _, task := range tasks {
		if task == recipe.TaskInference {
			continue
		}
		if _, duplicate := candidates[task]; duplicate {
			return nil, fmt.Errorf("model recipe: duplicate candidate task %q", task)
		}
		definition, err := CapabilityDefinition(task, resolved.Document.Model)
		if err != nil {
			return nil, err
		}
		candidates[task] = definition
	}
	return candidates, nil
}
