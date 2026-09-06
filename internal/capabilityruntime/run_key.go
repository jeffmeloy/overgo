package capabilityruntime

import (
	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// runKeyPrefix roots every capability run's workflow key.
const runKeyPrefix = "recipe/run/"

// RunKey keys a capability run on the recipe and the exact request
// document, so an identical request replays the same run.
func RunKey(definition recipe.Definition, request artifact.Content) string {
	return runKeyPrefix + definition.ID.String() + "/" + request.Descriptor.ID.String()
}
