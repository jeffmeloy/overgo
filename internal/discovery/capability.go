package discovery

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
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
	// KeyEnvironment names the variable a hosted model's provider key
	// lives in (empty for a local model), so a page can take the key when
	// the entry is refused for its absence.
	KeyEnvironment string
}

// CapabilityCatalog lists every model with any active recipe alias,
// across all tasks -- the full serving surface, where Servable is the
// inference slice. Aliases drive enumeration so capability-activated
// models (speech, forecast, tabular, projection) appear beside chat
// models without any per-model knowledge. The truncated result reports
// when the bounded alias listing could not carry every activation: a
// clipped catalog must say so rather than read as complete.
func CapabilityCatalog(ctx context.Context, store *overgodb.Store, limit int, memo *Memo) ([]CatalogEntry, bool, error) {
	return capabilityCatalog(ctx, store, limit, memo, false)
}

// RegisteredCatalog includes every registered model identity, even without an
// activation or available bytes. It shares the serving catalog's activation
// and presence checks; callers must refuse a truncated campaign denominator.
func RegisteredCatalog(ctx context.Context, store *overgodb.Store, limit int, memo *Memo) ([]CatalogEntry, bool, error) {
	return capabilityCatalog(ctx, store, limit, memo, true)
}

// CapabilityCatalogForTasks lists models activated for any selected task. An
// empty task selection preserves CapabilityCatalog's full-surface behavior.
// Filtering precedes presence verification to avoid hashing unrelated models.
func CapabilityCatalogForTasks(ctx context.Context, store *overgodb.Store, limit int, memo *Memo, tasks ...recipe.Task) ([]CatalogEntry, bool, error) {
	return capabilityCatalog(ctx, store, limit, memo, false, tasks...)
}

func capabilityCatalog(ctx context.Context, store *overgodb.Store, limit int, memo *Memo, includeInactive bool, tasks ...recipe.Task) ([]CatalogEntry, bool, error) {
	for _, task := range tasks {
		if !task.Valid() {
			return nil, false, fmt.Errorf("capability catalog: invalid task %q", task)
		}
	}
	// The alias projection is paged to its end: one bounded page shares its
	// budget with the recipe artifacts, so a store holding more recipes
	// than the page hid activations (the 27B's inference activation fell
	// outside the first page while its projection activation stayed in).
	// limit bounds the catalog entries, never the activations read.
	query := overgodb.Query{Kind: artifact.KindRecipe, MaxResults: limit, Projection: overgodb.ProjectAliases}
	var aliases []overgodb.AliasView
	for {
		result, err := store.Query(ctx, query)
		if err != nil {
			return nil, false, err
		}
		aliases = append(aliases, result.Aliases...)
		if result.Next == nil {
			break
		}
		query.Cursor = result.Next
	}
	tasksByModel := map[artifact.ID]map[recipe.Task]artifact.ID{}
	if includeInactive {
		registered, err := store.Query(ctx, overgodb.Query{
			Kind: artifact.KindModel, MaxResults: limit, Projection: overgodb.ProjectArtifacts,
		})
		if err != nil || registered.Truncated {
			return nil, registered.Truncated, err
		}
		for _, descriptor := range registered.Artifacts {
			tasksByModel[descriptor.ID] = map[recipe.Task]artifact.ID{}
		}
	}
	for _, alias := range aliases {
		model, task, ok := modelrecipe.ParseActiveAlias(alias.Name)
		if !ok {
			continue
		}
		if len(tasks) > 0 && !slices.Contains(tasks, task) {
			continue
		}
		if tasksByModel[model] == nil {
			if includeInactive {
				return nil, false, fmt.Errorf("discovery: activation names unregistered model %s", model)
			}
			tasksByModel[model] = map[recipe.Task]artifact.ID{}
		}
		tasksByModel[model][task] = alias.Target
	}
	models := slices.Collect(maps.Keys(tasksByModel))
	slices.SortFunc(models, artifact.CompareID)
	// The entry bound alone decides truncation, and a registered census
	// refuses a clipped denominator rather than reading as complete.
	truncated := len(models) > limit
	if truncated {
		if includeInactive {
			return nil, true, nil
		}
		models = models[:limit]
	}
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
		tasks := slices.Collect(maps.Keys(tasksByModel[model]))
		slices.Sort(tasks)
		for _, task := range tasks {
			capability := Capability{Task: task, Recipe: tasksByModel[model][task]}
			activation, active, err := modelrecipe.ActiveRecord(ctx, store, model, task)
			switch {
			case err != nil:
				capability.Stale = fmt.Sprintf("activation cannot be trusted: %v", err)
			case !active:
				capability.Stale = "declared activation is absent"
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
				// A remote model's inference lists with its provider's
				// refusal: the key its declaration names is absent.
				if task == recipe.TaskInference && found {
					if provider, remote, err := remoteprovider.Resolve(ctx, store, manifest); err != nil {
						capability.Stale = fmt.Sprintf("not servable: %v", err)
					} else if remote {
						entry.KeyEnvironment = provider.KeyEnvironment
						if refusal := remoteprovider.Refusal(provider); refusal != "" {
							capability.Stale = "not servable: " + refusal
						}
					}
				}
			}
			entry.Capabilities = append(entry.Capabilities, capability)
		}
		if len(entry.Capabilities) == 0 && !includeInactive {
			continue
		}
		entries = append(entries, entry)
	}
	return entries, truncated, nil
}
