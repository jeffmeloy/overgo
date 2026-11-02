package modelrecipe

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
	"overgo/internal/recipe"
)

// Derivation is what a program derived from its file holds that the store does not: what a record of its run names.
type Derivation struct {
	Inventory  modelartifact.Inventory
	Resolved   ResolvedModelDefinition
	Definition recipe.Definition
}

// PublishDerivation publishes the model's resolved facts and, when its recipe is not yet published, the recipe as a
// candidate: the model is then known to the store and no more.
func PublishDerivation(ctx context.Context, store artifact.Repository, derivation Derivation) error {
	// Facts already published are no change, and a registration repeated
	// for a model the store knows is a registration, not a fault.
	if err := PublishResolvedModelDefinition(ctx, store, derivation.Inventory, derivation.Resolved); err != nil {
		return err
	}
	_, published, err := Status(ctx, store, derivation.Definition.ID)
	if err != nil || published {
		return err
	}
	_, _, err = PublishCandidate(ctx, store, "recipe/candidate/"+derivation.Definition.ID.String(), derivation.Definition)
	return err
}

// DeriveInference derives what a recipe would hold of the model in file from the file alone: its definition under
// profile, the prototype its architecture selects, and the recipe binding it to the decode session its compiled plan
// allows, or to override where one is named. It publishes nothing.
func DeriveInference(
	file *gguf.File,
	inventory modelartifact.Inventory,
	spec model.Spec,
	profile ProfileDocument,
	override *DecodeSessionPolicy,
	residency recipe.ResidencyPolicy,
) (recipe.Definition, ResolvedModelDefinition, error) {
	fail := func(err error) (recipe.Definition, ResolvedModelDefinition, error) {
		return recipe.Definition{}, ResolvedModelDefinition{}, err
	}
	document, err := NewModelDefinitionDocument(profile, inventory.TensorInventory, spec)
	if err != nil {
		return fail(err)
	}
	resolved, err := document.Resolve(profile, inventory.TensorInventory)
	if err != nil {
		return fail(err)
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return fail(err)
	}
	plan, err := model.CompileModelPlanWithProfile(spec, weights, profile.Policy)
	if err != nil {
		return fail(err)
	}
	session := DecodeSessionCapacity
	if !plan.SupportsCapacityCache() {
		session = DecodeSessionRequest
	}
	if override != nil {
		if *override == DecodeSessionCapacity && !plan.SupportsCapacityCache() {
			return fail(errors.New("model recipe: the capacity session is unsupported by this model's compiled plan"))
		}
		session = *override
	}
	definition, err := InferenceWithModelDefinition(
		inventory.Manifest.ID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid, session, residency,
	)
	if err != nil {
		return fail(err)
	}
	return definition, resolved, nil
}

// bindDerived binds the program of a model no trusted recipe names, from its own file and the prototype compiled for
// its architecture: nothing authored or stored is needed to run it, and the state it carries says so.
func (l *LoadedProgram) bindDerived(ctx context.Context, store artifact.Reader) error {
	spec, err := model.ReadSpec(l.state.File)
	if err != nil {
		return err
	}
	profile, err := NewProfileDocument(spec.Profile())
	if err != nil {
		return err
	}
	definition, resolved, err := DeriveInference(l.state.File, l.state.Inventory, spec, profile, nil, recipe.ResidencyHybridNative)
	if err != nil {
		return err
	}
	policy, found, err := CatalogRuntimePolicy(definition.Task)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("model recipe: the runtime policy catalog holds no policy for inference")
	}
	if err := l.declarePolicy(ctx, store, policy); err != nil {
		return err
	}
	derivation := Derivation{Inventory: l.state.Inventory, Resolved: resolved, Definition: definition}
	if err := l.bindResolved(definition, resolved); err != nil {
		return err
	}
	l.state.Derived = &derivation
	return nil
}

// activationPage is the aliases refuseReplaced reads at a time.
const activationPage = 256

// activationLister is a store's listing of its activations, which a plain reader does not offer.
type activationLister interface {
	Query(context.Context, overgodb.Query) (overgodb.QueryResult, error)
}

// refuseReplaced refuses a file whose path an activation records for other bytes: weights that do not match their
// recorded hash, which no derivation puts right. A path merely remembered, or that nothing activated, runs as what it
// holds now.
func (l *LoadedProgram) refuseReplaced(ctx context.Context, store artifact.Reader) error {
	lister, listed := store.(activationLister)
	if !listed {
		return nil
	}
	path, err := pathidentity.Canonical(l.state.Path)
	if err != nil {
		return err
	}
	query := overgodb.Query{Kind: artifact.KindRecipe, MaxResults: activationPage, Projection: overgodb.ProjectAliases}
	for {
		result, err := lister.Query(ctx, query)
		if err != nil {
			return err
		}
		for _, alias := range result.Aliases {
			modelID, _, active := ParseActiveAlias(alias.Name)
			if !active || modelID == l.state.Inventory.Manifest.ID {
				continue
			}
			recorded, err := recordsPath(ctx, store, modelID, path)
			if err != nil {
				return err
			}
			if recorded {
				return fmt.Errorf("model recipe: %s is recorded as %s and now holds %s; restore its bytes, or move them to a path of their own",
					path, modelID, l.state.Inventory.Manifest.ID)
			}
		}
		if result.Next == nil {
			return nil
		}
		query.Cursor = result.Next
	}
}

// recordsPath reports whether the store records path as a location of the model's own bytes.
func recordsPath(ctx context.Context, store artifact.Reader, modelID artifact.ID, path string) (bool, error) {
	manifest, found, err := store.Manifest(ctx, modelID)
	if err != nil || !found {
		return false, err
	}
	for _, component := range manifest.Components {
		locations, err := store.Locations(ctx, component.Artifact)
		if err != nil {
			return false, err
		}
		for _, location := range locations {
			local := location.Kind == artifact.LocationFile || location.Kind == artifact.LocationDirectory
			if local && strings.EqualFold(filepath.ToSlash(filepath.Clean(location.Value)), filepath.ToSlash(path)) {
				return true, nil
			}
		}
	}
	return false, nil
}

// Derived returns what the program was derived from before transfer, and whether it was: a program bound from a stored
// recipe has none.
func (l *LoadedProgram) Derived() (Derivation, bool) {
	if l == nil || l.state == nil || l.state.Derived == nil {
		return Derivation{}, false
	}
	return *l.state.Derived, true
}
