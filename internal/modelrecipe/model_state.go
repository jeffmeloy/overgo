package modelrecipe

import (
	"context"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
)

// VerificationState names what the store records of a model's verification for a task.
type VerificationState string

const (
	// StateVerified is an activation whose evidence verifies against the compiled architecture profile.
	StateVerified VerificationState = "verified"
	// StateNotVerified is a model nothing has verified: no active recipe, whatever else the store holds of it.
	StateNotVerified VerificationState = "not verified"
	// StateOutOfDate is an activation whose evidence verifies against an architecture profile the build has moved
	// past, so what was measured is not quite what would run.
	StateOutOfDate VerificationState = "evidence out of date"
	// StateBroken is an activation the store cannot stand behind: its evidence does not verify, or its recipe was
	// refused or can no longer be read.
	StateBroken VerificationState = "verification broken"
)

// ModelState is a model's verification as a fact of its run and its listing, beside the evidence tier. It decides
// nothing about whether the model runs: verification marks what was measured, and Reason says what stands in its way.
type ModelState struct {
	Verification VerificationState
	Reason       string
}

// ReadModelState reads modelID's state for task from what the store records, with the activation that state rests on:
// the activation is set exactly when the state is StateVerified or StateOutOfDate. A record that cannot be
// trusted is a state and not a failure, so one broken activation leaves the model listed and runnable; only a canceled
// ctx, or a build that cannot state its own profiles, returns an error.
func ReadModelState(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (Activation, ModelState, error) {
	activation, active, err := ActiveRecord(ctx, store, modelID, task)
	switch {
	case ctx.Err() != nil:
		return Activation{}, ModelState{}, ctx.Err()
	case err != nil:
		return Activation{}, ModelState{StateBroken, err.Error()}, nil
	case !active:
		return Activation{}, ModelState{Verification: StateNotVerified}, nil
	}
	// Only a recipe bound to a model definition binds an architecture profile: an audio contract or a media source is a
	// profile of its own kind, which no build compiles.
	profile, bound := activation.Definition.PrimaryDependency(recipe.DependencyProfile)
	_, defined := activation.Definition.PrimaryDependency(recipe.DependencyDefinition)
	if !bound || !defined {
		return activation, ModelState{Verification: StateVerified}, nil
	}
	compiled, err := compiledProfiles()
	if err != nil {
		return Activation{}, ModelState{}, err
	}
	if !compiled[profile] {
		return activation, ModelState{StateOutOfDate, "verified against an architecture profile this build no longer compiles"}, nil
	}
	return activation, ModelState{Verification: StateVerified}, nil
}

// compiledProfiles is the identity of every architecture profile this build compiles.
var compiledProfiles = sync.OnceValues(func() (map[artifact.ID]bool, error) {
	compiled := map[artifact.ID]bool{}
	for _, architecture := range model.SupportedArchitectures() {
		registered, _ := model.LookupArchitecture(architecture)
		document, err := NewProfileDocument(registered)
		if err != nil {
			return nil, err
		}
		compiled[document.ID] = true
	}
	return compiled, nil
})
