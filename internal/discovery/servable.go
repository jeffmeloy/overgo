// Package discovery owns the servable predicate (floor component 9): a
// servable model is a model manifest with an ACTIVE inference recipe whose
// bytes are PRESENT at a recorded location. The smoke lane, the discovery
// query, and the evidence tier all consume this one predicate, so discovery
// cannot drift from recipe truth.
package discovery

import (
	"context"
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
}

// Servable lists models with active inference recipes; presence is a stat of
// recorded locations (manifest first, then components; recorded-but-missing
// or unrecorded reports Present=false, honestly).
func Servable(ctx context.Context, store *repodb.Store, limit int) ([]Entry, error) {
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindModel, MaxResults: limit})
	if err != nil {
		return nil, err
	}
	var entries []Entry
	for _, manifest := range result.Manifests {
		location, present := presence(ctx, store, manifest)
		activation, active, err := modelrecipe.ActiveRecord(ctx, store, manifest.ID, recipe.TaskInference)
		if err != nil {
			return nil, fmt.Errorf(
				"discovery: model %s activation at %q (present=%t): %w",
				manifest.ID, location, present, err,
			)
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

// presence prefers a location that stats as a REGULAR FILE (the artifact
// itself) over one that stats as a directory (often a recorded parent):
// serving needs the artifact path, not its neighborhood.
func presence(ctx context.Context, store *repodb.Store, manifest artifact.Manifest) (string, bool) {
	ids := []artifact.ID{manifest.ID}
	for _, component := range manifest.Components {
		ids = append(ids, component.Artifact)
	}
	recorded, directoryHit := "", ""
	for _, id := range ids {
		locations, err := store.Locations(ctx, id)
		if err != nil {
			continue
		}
		for _, location := range locations {
			if location.Kind != artifact.LocationFile && location.Kind != artifact.LocationDirectory {
				continue
			}
			recorded = location.Value
			info, err := os.Stat(location.Value)
			if err != nil {
				continue
			}
			if !info.IsDir() {
				return location.Value, true
			}
			if directoryHit == "" {
				directoryHit = location.Value
			}
		}
	}
	if directoryHit != "" {
		return directoryHit, true
	}
	return recorded, false
}
