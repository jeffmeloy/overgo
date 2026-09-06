package remoteprovider

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// Declaration is what one provider declaration committed: the provider
// document, the model manifest over it, the remote location the model is
// served by, and its activated inference recipe.
type Declaration struct {
	Provider Provider
	Model    artifact.ID
	Location string
	Recipe   recipe.Definition
}

// Declare commits the provider document, a model manifest whose one
// component is that document recorded at the remote location, and the
// remote inference recipe over the model, then activates the recipe. The
// activation's evidence is the declaration itself: a gate record whose
// one step names the declaration under the remote environment, at the
// experimental tier, since nothing about the hosted model is verified or
// reproducible from the store. A provider declared before returns the
// same identities without a second activation.
func Declare(ctx context.Context, store artifact.Repository, provider Provider, codeCommit string) (Declaration, error) {
	started := time.Now()
	declared, err := New(provider)
	if err != nil {
		return Declaration{}, err
	}
	content, err := codec.Content(declared)
	if err != nil {
		return Declaration{}, err
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentConfig, Name: "provider", Artifact: declared.ID,
	}})
	if err != nil {
		return Declaration{}, err
	}
	location := Location(declared)
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:       "remote-provider/declare/" + declared.ID.String(),
		Contents:  []artifact.Content{content},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{
			Artifact: declared.ID, Kind: artifact.LocationRemote, Value: location,
		}, Action: artifact.LocationAdd}},
	}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return Declaration{}, err
	}
	definition, err := modelrecipe.RemoteInferenceDefinition(manifest.ID, declared.ID)
	if err != nil {
		return Declaration{}, err
	}
	declaration := Declaration{Provider: declared, Model: manifest.ID, Location: location, Recipe: definition}
	if activation, active, err := modelrecipe.ActiveRecord(ctx, store, manifest.ID, recipe.TaskInference); err != nil {
		return Declaration{}, err
	} else if active && activation.Definition.ID == definition.ID {
		return declaration, nil
	}
	// The declaration record names the recipe as its lineage, so the
	// recipe enters the store as a candidate first.
	if _, published, err := modelrecipe.Status(ctx, store, definition.ID); err != nil {
		return Declaration{}, err
	} else if !published {
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "remote-provider/candidate/"+definition.ID.String(), definition); err != nil {
			return Declaration{}, err
		}
	}
	environment, err := Environment(declared)
	if err != nil {
		return Declaration{}, err
	}
	environmentContent, err := environment.Content()
	if err != nil {
		return Declaration{}, err
	}
	// The declaration's own duration is the step's measure; a record needs a
	// positive one, and the smallest positive duration bounds a clock that
	// did not advance.
	duration := max(uint64(time.Since(started).Nanoseconds()), 1)
	record, err := runrecord.NewGateRecord(
		definition.ID, environment.ID, codeCommit, runrecord.OutcomeSucceeded, "", duration,
		[]runrecord.GateStep{{
			Name: "remote-provider-declaration", Phase: runrecord.PhaseLoad,
			Outcome: runrecord.StepSucceeded, DurationNS: duration,
		}},
	)
	if err != nil {
		return Declaration{}, err
	}
	batch, err := record.Batch("remote-provider/declaration/" + definition.ID.String() + "/" + record.Result.ID.String())
	if err != nil {
		return Declaration{}, err
	}
	batch.Contents = append(batch.Contents, environmentContent)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return Declaration{}, err
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID},
		recipe.EvidenceExperimental, "remote provider declared; the hosted model is not reproducible from the store",
	); err != nil {
		return Declaration{}, fmt.Errorf("remote provider: activate %s: %w", location, err)
	}
	return declaration, nil
}

// Resolve reads the provider a model manifest declares; remote is false
// for a local model.
func Resolve(ctx context.Context, reader artifact.Reader, manifest artifact.Manifest) (Provider, bool, error) {
	for _, component := range manifest.Components {
		if component.Role != artifact.ComponentConfig || component.Artifact.Kind() != artifact.KindProfile {
			continue
		}
		content, found, err := artifact.ReadContent(ctx, reader, component.Artifact)
		if err != nil {
			return Provider{}, false, err
		}
		if !found || content.Descriptor.MediaType != MediaType {
			continue
		}
		provider, err := codec.Parse(content.Data)
		if err != nil {
			return Provider{}, false, err
		}
		return provider, true, nil
	}
	return Provider{}, false, nil
}

// Declared is one remote model the store holds: its provider, its model
// identity and the location it is served by.
type Declared struct {
	Provider Provider
	Model    artifact.ID
	Location string
}

// List reads every remote model the store declares.
func List(ctx context.Context, store *overgodb.Store, limit int) ([]Declared, error) {
	result, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindModel, MaxResults: limit, Projection: overgodb.ProjectManifests,
	})
	if err != nil {
		return nil, err
	}
	if result.Truncated {
		return nil, fmt.Errorf("remote provider: model catalog exceeds listing bound %d", limit)
	}
	var declared []Declared
	for _, manifest := range result.Manifests {
		provider, remote, err := Resolve(ctx, store, manifest)
		if err != nil {
			return nil, err
		}
		if remote {
			declared = append(declared, Declared{Provider: provider, Model: manifest.ID, Location: Location(provider)})
		}
	}
	return declared, nil
}

// Reference resolves a serving reference, the remote location or the
// model identity, to the remote model the store declares; remote is false
// when no declared provider serves the reference, which is then a local
// model path.
func Reference(ctx context.Context, store *overgodb.Store, limit int, reference string) (Provider, artifact.ID, bool, error) {
	declared, err := List(ctx, store, limit)
	if err != nil {
		return Provider{}, artifact.ID{}, false, err
	}
	for _, model := range declared {
		if model.Location == reference || model.Model.String() == reference {
			return model.Provider, model.Model, true, nil
		}
	}
	return Provider{}, artifact.ID{}, false, nil
}
