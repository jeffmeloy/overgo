// Package discovery owns the servable predicate (floor component 9): a
// servable model is a model manifest with an ACTIVE inference recipe whose
// bytes are PRESENT at a recorded location. The smoke lane, the discovery
// query, and the evidence tier all consume this one predicate, so discovery
// cannot drift from recipe truth.
package discovery

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
)

type Entry struct {
	Model    artifact.ID
	Recipe   artifact.ID
	Tier     recipe.EvidenceTier
	Location string
	Present  bool
	// Stale carries the activation defect when the active recipe cannot be
	// trusted. A stale entry is reported, never served: one broken
	// activation must not blind discovery to every healthy model.
	Stale string
}

// Servable lists models with active inference recipes; presence is a stat of
// recorded locations (manifest first, then components; recorded-but-missing
// or unrecorded reports Present=false).
func Servable(ctx context.Context, store *overgodb.Store, limit int) ([]Entry, error) {
	return ServableWithMemo(ctx, store, limit, nil)
}

// ServableWithMemo is Servable with digest reuse for interactive callers: a
// nil memo hashes every file fresh, exactly as Servable always has.
// Explicit locations filter recorded component metadata before any file hashing
// or active-recipe resolution. An empty selection retains full-catalog discovery.
func ServableWithMemo(ctx context.Context, store *overgodb.Store, limit int, memo *Memo, selectedLocations ...string) ([]Entry, error) {
	result, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindModel, MaxResults: limit, Projection: overgodb.ProjectManifests,
	})
	if err != nil {
		return nil, err
	}
	if result.Truncated {
		return nil, fmt.Errorf("discovery: model catalog exceeds listing bound %d", limit)
	}
	var entries []Entry
	identities := map[string]fileIdentity{}
	for _, manifest := range result.Manifests {
		selected, err := matchesSelectedLocation(ctx, store, manifest, selectedLocations)
		if err != nil {
			return nil, err
		}
		if !selected {
			continue
		}
		declared, err := modelrecipe.HasActiveRecipe(ctx, store, manifest.ID, recipe.TaskInference)
		if err != nil {
			return nil, err
		}
		if !declared {
			continue
		}
		location, present := presence(ctx, store, manifest, identities, memo)
		if !present {
			continue
		}
		activation, active, err := modelrecipe.ActiveRecord(ctx, store, manifest.ID, recipe.TaskInference)
		if err != nil {
			entries = append(entries, Entry{
				Model: manifest.ID, Location: location, Present: present,
				Stale: fmt.Sprintf("activation cannot be trusted: %v", err),
			})
			continue
		}
		if !active {
			continue
		}
		entry := Entry{
			Model: manifest.ID, Recipe: activation.Definition.ID, Tier: activation.Tier,
			Location: location, Present: present,
		}
		// The loader refuses an inference recipe without its runtime
		// policy, so the servable listing must report it stale exactly as
		// the capability catalog does -- eval and swap targets derived
		// from this listing never include a model the loader refuses.
		if _, err := modelrecipe.ResolveRuntimePolicy(ctx, store, activation.Definition); err != nil {
			entry.Stale = fmt.Sprintf("not servable: %v (recipe policy <model> binds it)", err)
		}
		// A remote model lists with its provider's refusal: the key its
		// declaration names is absent from the environment.
		if provider, remote, err := remoteprovider.Resolve(ctx, store, manifest); err != nil {
			entry.Stale = fmt.Sprintf("not servable: %v", err)
		} else if refusal := remoteprovider.Refusal(provider); remote && refusal != "" {
			entry.Stale = "not servable: " + refusal
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func matchesSelectedLocation(ctx context.Context, store artifact.Reader, manifest artifact.Manifest, selected []string) (bool, error) {
	if len(selected) == 0 {
		return true, nil
	}
	for _, component := range manifest.Components {
		locations, err := store.Locations(ctx, component.Artifact)
		if err != nil {
			return false, err
		}
		for _, location := range locations {
			if location.Kind != artifact.LocationFile {
				continue
			}
			for _, path := range selected {
				if strings.EqualFold(filepath.ToSlash(filepath.Clean(location.Value)), filepath.ToSlash(filepath.Clean(path))) {
					return true, nil
				}
			}
		}
	}
	return false, nil
}

type fileIdentity struct {
	id      artifact.ID
	size    uint64
	present bool
}

// presence requires recorded bytes, not a path that now names replacement bytes.
func presence(
	ctx context.Context,
	store *overgodb.Store,
	manifest artifact.Manifest,
	identities map[string]fileIdentity,
	memo *Memo,
) (string, bool) {
	recorded, servingLocation := "", ""
	for _, component := range manifest.Components {
		descriptor, ok, err := store.Artifact(ctx, component.Artifact)
		if err != nil || !ok {
			return recorded, false
		}
		locations, err := store.Locations(ctx, component.Artifact)
		if err != nil {
			return recorded, false
		}
		matched := false
		for _, location := range locations {
			// A remote model's bytes live with its provider: the remote
			// location is its serving location and its presence.
			if location.Kind == artifact.LocationRemote {
				return location.Value, true
			}
			if location.Kind != artifact.LocationFile {
				continue
			}
			if recorded == "" {
				recorded = location.Value
			}
			identity := identifyLocation(location.Value, descriptor.ID.Kind(), identities, memo)
			if identity.present && identity.id == descriptor.ID && identity.size == descriptor.Size {
				matched = true
				if servingLocation == "" {
					servingLocation = location.Value
				}
				break
			}
		}
		if !matched {
			return recorded, false
		}
	}
	if servingLocation != "" {
		return servingLocation, true
	}
	if recorded != "" {
		return recorded, false
	}
	locations, _ := store.Locations(ctx, manifest.ID)
	for _, location := range locations {
		if location.Kind == artifact.LocationDirectory {
			return location.Value, false
		}
	}
	return "", false
}

// WeightsIdentity is the model identity of the weights file at path,
// the digest a capability claim binds to (catalogs identify manifests;
// claims identify the bytes they measured, exactly as the benchmark
// publisher does), through the memo so a file hashed for the listing
// is not hashed again. A path that does not name present bytes returns
// an error.
func WeightsIdentity(path string, memo *Memo) (artifact.ID, error) {
	identity := identifyLocation(path, artifact.KindModel, map[string]fileIdentity{}, memo)
	if !identity.present {
		return artifact.ID{}, fmt.Errorf("discovery: %s does not name present weights", path)
	}
	return identity.id, nil
}

func identifyLocation(path string, kind artifact.Kind, identities map[string]fileIdentity, memo *Memo) fileIdentity {
	key := path + "\x00" + kind.String()
	if identity, ok := identities[key]; ok {
		return identity
	}
	identity := fileIdentity{}
	before, err := os.Stat(path)
	if err != nil || before.IsDir() {
		identities[key] = identity
		return identity
	}
	if remembered, ok := memo.lookup(key, before); ok {
		identities[key] = remembered
		return remembered
	}
	file, err := os.Open(path)
	if err != nil {
		identities[key] = identity
		return identity
	}
	id, size, identifyErr := artifact.Identify(kind, file)
	closeErr := file.Close()
	after, statErr := os.Stat(path)
	if errors.Join(identifyErr, closeErr, statErr) == nil &&
		before.Size() == after.Size() && before.ModTime().Equal(after.ModTime()) && size == uint64(after.Size()) {
		identity = fileIdentity{id: id, size: size, present: true}
	}
	if identity.present {
		memo.record(key, after, identity)
	}
	identities[key] = identity
	return identity
}
