package discovery

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// ActiveProjector resolves the projector bytes a model's active projection
// recipe binds: the file the server loads when its command line names no
// projector, so every launcher (the swap proxy, the desktop launcher, the
// browser lane) serves the modalities the store declares for the model. A
// model without a projection activation serves text only (ok false, no
// error). An activation whose projector bytes are absent or differ from
// the recorded artifact is refused, never served: presence means the
// recorded bytes, as it does for the model itself.
func ActiveProjector(ctx context.Context, store *overgodb.Store, modelID artifact.ID, memo *Memo) (string, bool, error) {
	_, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskProjection)
	if err != nil {
		return "", false, err
	}
	if !active {
		return "", false, nil
	}
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskProjection)
	if err != nil {
		return "", false, err
	}
	projectorID, ok := program.Definition().PrimaryDependency(recipe.DependencyProjector)
	if !ok {
		return "", false, errors.New("discovery: the active projection recipe binds no projector")
	}
	manifest, found, err := store.Manifest(ctx, projectorID)
	if err != nil {
		return "", false, err
	}
	if !found {
		return "", false, fmt.Errorf("discovery: projector %s has no manifest in the store", projectorID)
	}
	location, present := presence(ctx, store, manifest, map[string]fileIdentity{}, memo)
	if !present {
		return "", false, fmt.Errorf("discovery: projector %s for model %s has no recorded bytes on disk (recorded location %q)", projectorID, modelID, location)
	}
	return location, true, nil
}
