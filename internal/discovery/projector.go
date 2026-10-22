package discovery

import (
	"context"
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
	return activeComponent(ctx, store, modelID, recipe.TaskProjection, recipe.DependencyProjector, memo)
}

// ActiveDraft resolves the draft bytes a model's active speculation recipe
// binds, as ActiveProjector resolves a projector's: the draft the server
// loads when its command line names none, so every launcher speculates
// through the draft the store declares for the model. A model without a
// speculation activation decodes a token a step (ok false, no error).
func ActiveDraft(ctx context.Context, store *overgodb.Store, modelID artifact.ID, memo *Memo) (string, bool, error) {
	return activeComponent(ctx, store, modelID, recipe.TaskSpeculation, recipe.DependencyDraft, memo)
}

// activeComponent resolves the bytes of the role a model's active recipe for task binds.
func activeComponent(
	ctx context.Context, store *overgodb.Store, modelID artifact.ID, task recipe.Task, role recipe.DependencyRole, memo *Memo,
) (string, bool, error) {
	_, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, task)
	if err != nil {
		return "", false, err
	}
	if !active {
		return "", false, nil
	}
	_, program, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, task)
	if err != nil {
		return "", false, err
	}
	componentID, ok := program.Definition().PrimaryDependency(role)
	if !ok {
		return "", false, fmt.Errorf("discovery: the active %s recipe binds no %s", task, role)
	}
	manifest, found, err := store.Manifest(ctx, componentID)
	if err != nil {
		return "", false, err
	}
	if !found {
		return "", false, fmt.Errorf("discovery: %s %s has no manifest in the store", role, componentID)
	}
	location, present := presence(ctx, store, manifest, map[string]fileIdentity{}, memo)
	if !present {
		return "", false, fmt.Errorf("discovery: %s %s for model %s has no recorded bytes on disk (recorded location %q)", role, componentID, modelID, location)
	}
	return location, true, nil
}
