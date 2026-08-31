package discovery

import (
	"context"
	"fmt"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// Capability reports one task activation on a catalogued model. Stale
// carries the activation defect when the record cannot be trusted; a
// stale capability is reported, never served.
type Capability struct {
	Task   recipe.Task
	Recipe artifact.ID
	Tier   recipe.EvidenceTier
	Stale  string
}

// CatalogEntry is one model with every task its activation aliases
// declare. Presence is the same recorded-bytes predicate Servable uses.
type CatalogEntry struct {
	Model        artifact.ID
	Location     string
	Present      bool
	Capabilities []Capability
}

// CapabilityCatalog lists every model with any active recipe alias,
// across all tasks -- the full serving surface, where Servable is the
// inference slice. Aliases drive enumeration so capability-activated
// models (speech, forecast, tabular, projection) appear beside chat
// models without any per-model knowledge. The truncated result reports
// when the bounded alias listing could not carry every activation: a
// clipped catalog must say so rather than read as complete.
func CapabilityCatalog(ctx context.Context, store *overgodb.Store, limit int, memo *Memo) ([]CatalogEntry, bool, error) {
	result, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindRecipe, MaxResults: limit, Projection: overgodb.ProjectAliases,
	})
	if err != nil {
		return nil, false, err
	}
	tasksByModel := map[artifact.ID][]recipe.Task{}
	for _, alias := range result.Aliases {
		model, task, ok := modelrecipe.ParseActiveAlias(alias.Name)
		if !ok {
			continue
		}
		tasksByModel[model] = append(tasksByModel[model], task)
	}
	models := make([]artifact.ID, 0, len(tasksByModel))
	for model := range tasksByModel {
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].String() < models[j].String() })
	entries := make([]CatalogEntry, 0, len(models))
	identities := map[string]fileIdentity{}
	for _, model := range models {
		entry := CatalogEntry{Model: model}
		manifest, found, err := store.Manifest(ctx, model)
		if err != nil {
			return nil, false, err
		}
		if found {
			entry.Location, entry.Present = presence(ctx, store, manifest, identities, memo)
		} else if path, pathErr := artifact.AvailablePath(ctx, store, model, artifact.LocationFile); pathErr == nil {
			// Models recorded through location records rather than piece
			// manifests — safetensors directories and single-file weights —
			// are present when a recorded location still resolves on disk.
			entry.Location, entry.Present = path, true
		} else if path, pathErr := artifact.AvailablePath(ctx, store, model, artifact.LocationDirectory); pathErr == nil {
			entry.Location, entry.Present = path, true
		}
		tasks := tasksByModel[model]
		sort.Slice(tasks, func(i, j int) bool { return tasks[i] < tasks[j] })
		for _, task := range tasks {
			capability := Capability{Task: task}
			activation, active, err := modelrecipe.ActiveRecord(ctx, store, model, task)
			switch {
			case err != nil:
				capability.Stale = fmt.Sprintf("activation cannot be trusted: %v", err)
			case !active:
				continue
			default:
				capability.Recipe, capability.Tier = activation.Definition.ID, activation.Tier
				// Servability is the serving stack's own gate: for tasks the
				// policy catalog supports, loading refuses a recipe without a
				// bound runtime policy, so the catalog must report it stale
				// rather than list it as launchable. Tasks outside the policy
				// catalog carry no policy requirement -- the same optional
				// resolution the capability selector performs, no policy copy.
				if _, supported, err := modelrecipe.CatalogRuntimePolicy(task); err != nil {
					capability.Stale = fmt.Sprintf("runtime policy catalog cannot be read: %v", err)
				} else if supported {
					if _, err := modelrecipe.ResolveRuntimePolicy(ctx, store, activation.Definition); err != nil {
						capability.Stale = fmt.Sprintf("not servable: %v (recipe policy <model> binds it)", err)
					}
				}
			}
			entry.Capabilities = append(entry.Capabilities, capability)
		}
		if len(entry.Capabilities) == 0 {
			continue
		}
		entries = append(entries, entry)
	}
	return entries, result.Truncated, nil
}
