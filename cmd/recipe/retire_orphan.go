package main

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// retireOrphan retires an activation whose own trust check fails: the
// positional argument is the model artifact ID the stale alias binds,
// because an orphaned identity no longer resolves from any file.
func retireOrphan(repository, modelID string, task recipe.Task, reason string) error {
	model, err := artifact.ParseID(modelID)
	if err != nil {
		return errors.Join(errors.New("retire-orphan takes the model artifact ID, not a path"), err)
	}
	revision, err := modelintake.CleanRevision(context.Background())
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	if err := modelrecipe.RetireOrphanedActivation(context.Background(), store, model, task, revision, reason); err != nil {
		return err
	}
	fmt.Printf("retired orphaned %s activation for %s\n", task, model)
	return nil
}
