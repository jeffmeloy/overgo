package modelrecipe

import (
	"context"
	"errors"
	"io"
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

// ReadModelState reads modelID's state for task from what the store records, with the activation that state rests on: it
// is whole when the state is StateVerified or StateOutOfDate, and holds only the Definition the alias names when the
// state is StateBroken and that recipe can still be read. A record that cannot be trusted is a state and not a failure,
// so one broken activation leaves the model listed and runnable. A store that fails to answer says nothing of the
// model's verification, so that, a canceled ctx, and a build that cannot state its own profiles return an error.
func ReadModelState(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (Activation, ModelState, error) {
	reads := &observedReader{Reader: store}
	activation, active, err := ActiveRecord(ctx, reads, modelID, task)
	switch {
	case ctx.Err() != nil:
		return Activation{}, ModelState{}, ctx.Err()
	case reads.failure != nil:
		return Activation{}, ModelState{}, reads.failure
	case err != nil:
		named, readable := aliasedDefinition(ctx, reads, modelID, task)
		if reads.failure != nil {
			return Activation{}, ModelState{}, reads.failure
		}
		if !readable {
			return Activation{}, ModelState{StateBroken, err.Error()}, nil
		}
		return Activation{Definition: named}, ModelState{StateBroken, err.Error()}, nil
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

// aliasedDefinition reads the recipe a model's active alias names, whatever its evidence says of it.
func aliasedDefinition(ctx context.Context, store artifact.Reader, modelID artifact.ID, task recipe.Task) (recipe.Definition, bool) {
	id, aliased, err := artifact.ResolveAlias(ctx, store, activeAlias(modelID, task))
	if err != nil || !aliased {
		return recipe.Definition{}, false
	}
	definition, err := recipe.RequireDefinition(ctx, store, id)
	if err != nil || definition.Model != modelID || definition.Task != task {
		return recipe.Definition{}, false
	}
	return definition, true
}

// observedReader keeps the first error the store itself answered with, so that a record which reads and does not stand
// can be told from a store that could not answer; the checks over it fold the two into one error.
type observedReader struct {
	artifact.Reader
	failure error
}

func answer[T any](r *observedReader, value T, err error) (T, error) {
	if err != nil && r.failure == nil {
		r.failure = err
	}
	return value, err
}

// Artifact answers as the store does, keeping a failure.
func (r *observedReader) Artifact(ctx context.Context, id artifact.ID) (artifact.Descriptor, bool, error) {
	descriptor, found, err := r.Reader.Artifact(ctx, id)
	_, err = answer(r, descriptor, err)
	return descriptor, found, err
}

// OpenContent answers as the store does, keeping a failure.
func (r *observedReader) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	descriptor, content, found, err := r.Reader.OpenContent(ctx, id)
	_, err = answer(r, descriptor, err)
	if err != nil || content == nil {
		return descriptor, content, found, err
	}
	return descriptor, observedContent{content, r}, found, nil
}

// Manifest answers as the store does, keeping a failure.
func (r *observedReader) Manifest(ctx context.Context, id artifact.ID) (artifact.Manifest, bool, error) {
	manifest, found, err := r.Reader.Manifest(ctx, id)
	_, err = answer(r, manifest, err)
	return manifest, found, err
}

// ResolveAlias answers as the store does, keeping a failure.
func (r *observedReader) ResolveAlias(ctx context.Context, name string) (artifact.ID, bool, error) {
	id, found, err := r.Reader.ResolveAlias(ctx, name)
	_, err = answer(r, id, err)
	return id, found, err
}

// Parents answers as the store does, keeping a failure.
func (r *observedReader) Parents(ctx context.Context, id artifact.ID) ([]artifact.Lineage, error) {
	lineage, err := r.Reader.Parents(ctx, id)
	return answer(r, lineage, err)
}

// Children answers as the store does, keeping a failure.
func (r *observedReader) Children(ctx context.Context, id artifact.ID) ([]artifact.Lineage, error) {
	lineage, err := r.Reader.Children(ctx, id)
	return answer(r, lineage, err)
}

// Locations answers as the store does, keeping a failure.
func (r *observedReader) Locations(ctx context.Context, id artifact.ID) ([]artifact.Location, error) {
	locations, err := r.Reader.Locations(ctx, id)
	return answer(r, locations, err)
}

// observedContent keeps a content stream's failure as its reader's; its end is no failure.
type observedContent struct {
	io.Reader
	reads *observedReader
}

// Read reads the stream, keeping a failure.
func (c observedContent) Read(data []byte) (int, error) {
	count, err := c.Reader.Read(data)
	if err != nil && !errors.Is(err, io.EOF) && c.reads.failure == nil {
		c.reads.failure = err
	}
	return count, err
}
