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
// models without any per-model knowledge.
func CapabilityCatalog(ctx context.Context, store *overgodb.Store, limit int, memo *Memo) ([]CatalogEntry, error) {
	result, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindRecipe, MaxResults: limit, Projection: overgodb.ProjectAliases,
	})
	if err != nil {
		return nil, err
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
			return nil, err
		}
		if found {
			entry.Location, entry.Present = presence(ctx, store, manifest, identities, memo)
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
			}
			entry.Capabilities = append(entry.Capabilities, capability)
		}
		if len(entry.Capabilities) == 0 {
			continue
		}
		entries = append(entries, entry)
	}
	return entries, nil
}
