package remoteprovider

import (
	"context"
	"errors"
	"fmt"
	"slices"
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
	duration := stepDuration(started)
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

// Document is a provider declaration as the CLI file and the library
// route carry it: one provider, the model ids it serves.
type Document struct {
	Name           string   `json:"name"`
	Endpoint       string   `json:"endpoint"`
	KeyEnvironment string   `json:"key_environment"`
	Models         []string `json:"models"`
	ContextLength  uint32   `json:"context_length,omitzero"`
}

// DeclareDocument declares every model of a document (each its own
// manifest and activation over the one provider) and returns the
// declarations in the document's order; a document naming no model is
// refused.
func DeclareDocument(ctx context.Context, store artifact.Repository, document Document, codeCommit string) ([]Declaration, error) {
	if len(document.Models) == 0 {
		return nil, errors.New("remote provider: the declaration names no model")
	}
	declarations := make([]Declaration, 0, len(document.Models))
	for _, model := range document.Models {
		declared, err := Declare(ctx, store, Provider{
			Name: document.Name, Endpoint: document.Endpoint, KeyEnvironment: document.KeyEnvironment,
			Model: model, ContextLength: document.ContextLength,
		}, codeCommit)
		if err != nil {
			return declarations, err
		}
		declarations = append(declarations, declared)
	}
	return declarations, nil
}

// stepDuration: step duration since started; floor 1ns (records need a
// positive duration; clock may not advance).
func stepDuration(started time.Time) uint64 {
	return max(uint64(time.Since(started).Nanoseconds()), 1)
}

// retirementStep: retirement step name + failed record's failure code;
// prose reason goes in step evidence.
const retirementStep = "remote-provider-retirement"

// Retire records a failed gate under the remote environment (reason =
// endpoint gone, key withdrawn) -> evidence-backed retirement -> alias
// released; declaration stays in history, catalog stops listing.
func Retire(ctx context.Context, store *overgodb.Store, limit int, location, codeCommit, reason string) error {
	started := time.Now()
	declared, err := List(ctx, store, limit)
	if err != nil {
		return err
	}
	index := slices.IndexFunc(declared, func(model Declared) bool { return model.Location == location })
	if index < 0 {
		return fmt.Errorf("remote provider: no declared model at %s", location)
	}
	model := declared[index]
	definition, err := modelrecipe.RemoteInferenceDefinition(model.Model, model.Provider.ID)
	if err != nil {
		return err
	}
	environment, err := Environment(model.Provider)
	if err != nil {
		return err
	}
	environmentContent, err := environment.Content()
	if err != nil {
		return err
	}
	duration := stepDuration(started)
	record, err := runrecord.NewGateRecord(
		definition.ID, environment.ID, codeCommit, runrecord.OutcomeFailed, retirementStep, duration,
		[]runrecord.GateStep{{
			Name: retirementStep, Phase: runrecord.PhaseLoad,
			Outcome: runrecord.StepFailed, DurationNS: duration, Evidence: reason,
		}},
	)
	if err != nil {
		return err
	}
	batch, err := record.Batch("remote-provider/retirement/" + definition.ID.String() + "/" + record.Result.ID.String())
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, environmentContent)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return err
	}
	if err := modelrecipe.RetireActiveCapability(ctx, store, definition, modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}, reason); err != nil {
		return err
	}
	// No successor: release the alias so the model leaves the catalog.
	return modelrecipe.ReleaseRetiredAlias(ctx, store, model.Model, recipe.TaskInference)
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

// List reads declared remote models with an active inference recipe;
// retired declarations stay in history, unlisted. The limit bounds active
// remote models, not the history scanned to find them.
func List(ctx context.Context, store *overgodb.Store, limit int) ([]Declared, error) {
	query := overgodb.Query{
		Kind: artifact.KindModel, MaxResults: limit, Projection: overgodb.ProjectManifests,
	}
	var declared []Declared
	for {
		result, err := store.Query(ctx, query)
		if err != nil {
			return nil, err
		}
		for _, manifest := range result.Manifests {
			provider, remote, err := Resolve(ctx, store, manifest)
			if err != nil {
				return nil, err
			}
			if !remote {
				continue
			}
			active, err := modelrecipe.HasActiveRecipe(ctx, store, manifest.ID, recipe.TaskInference)
			if err != nil {
				return nil, err
			}
			if active {
				if len(declared) == limit {
					return nil, fmt.Errorf("remote provider: active model catalog exceeds listing bound %d", limit)
				}
				declared = append(declared, Declared{Provider: provider, Model: manifest.ID, Location: Location(provider)})
			}
		}
		if result.Next == nil {
			return declared, nil
		}
		query.Cursor = result.Next
	}
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
