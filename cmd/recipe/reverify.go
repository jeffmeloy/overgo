package main

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// reverificationReason is the decision a re-verification on a model's recorded suite records.
const reverificationReason = "re-verified on the exact suite its accepted verification ran, for the recipe this build derives"

// reverifyRecorded re-verifies the model at path, whose active inference recipe this build no longer derives, on the
// exact suite its accepted verification ran (reverifyOn), and activates the recipe this build derives once every case
// passes: on 2026-11-08 five servable models, the judge among them, held evidence for an architecture profile their
// build no longer compiled, and opened all the same with a note. A failing case leaves the active recipe as it was.
func reverifyRecorded(repository, path string, override modelintake.SessionOverride, residency recipe.ResidencyPolicy) error {
	revision, err := modelintake.CleanRevision(context.Background())
	if err != nil {
		return err
	}
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	candidate, err := prepareExactCandidate(ctx, store, path, override, residency)
	if err != nil {
		return errors.Join(err, store.Close())
	}
	runtime := &deferredExactRuntime{open: func(ctx context.Context) (exactRuntime, error) {
		return openExactCandidate(ctx, store, path, candidate)
	}}
	verification, err := reverifyOn(ctx, store, candidate.Inventory.Manifest.ID, candidate.Definition, revision, candidate.Resolved.Document.ID, runtime)
	if err := errors.Join(err, store.Close()); err != nil {
		return err
	}
	return activate(repository, path, reverificationReason, override, residency, verification)
}

// reverifyOn runs, through definition, the recipe this build derives for model, the exact suite the model's active
// recipe of the same task was accepted on, read from the store (evaluation.RecordedExactSuite), and returns the
// verification it records. A model whose active recipe is the one this build derives has nothing to re-verify.
func reverifyOn(ctx context.Context, store *overgodb.Store, model artifact.ID, definition recipe.Definition, revision string, modelDefinition artifact.ID, runtime *deferredExactRuntime) (modelrecipe.Verification, error) {
	active, found, err := modelrecipe.ActiveRecord(ctx, store, model, definition.Task)
	if err != nil || !found {
		return modelrecipe.Verification{}, errors.Join(err, runtime.Close(), fmt.Errorf("recipe: %s has no active %s recipe to re-verify", model, definition.Task))
	}
	if active.Definition.ID == definition.ID {
		return modelrecipe.Verification{}, errors.Join(runtime.Close(), fmt.Errorf("recipe: the active recipe %s is the one this build derives: nothing to re-verify", definition.ID))
	}
	if len(active.Verified.Gate.Steps) != 1 {
		return modelrecipe.Verification{}, errors.Join(runtime.Close(), fmt.Errorf("recipe: the accepted verification %s of %s is no single exact suite", active.Verified.Gate.ID, active.Definition.ID))
	}
	suite, err := evaluation.RecordedExactSuite(ctx, store, active.Verified.Gate.Steps[0].Evidence)
	if err != nil {
		return modelrecipe.Verification{}, errors.Join(err, runtime.Close())
	}
	exactPlan, err := evaluation.CompileExact(suite)
	if err != nil {
		return modelrecipe.Verification{}, errors.Join(err, runtime.Close())
	}
	return verifyExactRuntime(ctx, store, definition, revision, suite, exactPlan, modelDefinition, runtime)
}
