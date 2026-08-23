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

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
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
// or unrecorded reports Present=false, honestly).
func Servable(ctx context.Context, store *repodb.Store, limit int) ([]Entry, error) {
	result, err := store.Query(ctx, repodb.Query{
		Kind: artifact.KindModel, MaxResults: limit, Projection: repodb.ProjectManifests,
	})
	if err != nil {
		return nil, err
	}
	var entries []Entry
	identities := map[string]fileIdentity{}
	for _, manifest := range result.Manifests {
		declared, err := modelrecipe.HasActiveRecipe(ctx, store, manifest.ID, recipe.TaskInference)
		if err != nil {
			return nil, err
		}
		if !declared {
			continue
		}
		location, present := presence(ctx, store, manifest, identities)
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
		entries = append(entries, Entry{
			Model: manifest.ID, Recipe: activation.Definition.ID, Tier: activation.Tier,
			Location: location, Present: present,
		})
	}
	return entries, nil
}

type fileIdentity struct {
	id      artifact.ID
	size    uint64
	present bool
}

// presence requires recorded bytes, not a path that now names replacement bytes.
func presence(
	ctx context.Context,
	store *repodb.Store,
	manifest artifact.Manifest,
	identities map[string]fileIdentity,
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
			if location.Kind != artifact.LocationFile {
				continue
			}
			if recorded == "" {
				recorded = location.Value
			}
			identity := identifyLocation(location.Value, descriptor.ID.Kind(), identities)
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

func identifyLocation(path string, kind artifact.Kind, identities map[string]fileIdentity) fileIdentity {
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
	identities[key] = identity
	return identity
}
