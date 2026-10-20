package main

import (
	"errors"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// mediaPending lists the media capabilities and inactive media recipes that
// another lane put into the shared store before the frozen image-video
// inventory recorded them. The inventory's acceptance sets exactly these
// aside, under the finding that names them, until their owner freezes them.
type mediaPending struct {
	Finding  string               `json:"finding"`
	Owner    string               `json:"owner"`
	Reason   string               `json:"reason"`
	Cells    []mediaPendingCell   `json:"cells"`
	Inactive []imageVideoInactive `json:"inactive"`
}

type mediaPendingCell struct {
	Model  artifact.ID `json:"model"`
	Task   recipe.Task `json:"task"`
	Recipe artifact.ID `json:"recipe"`
}

func loadMediaPending(root string) (mediaPending, error) {
	var value mediaPending
	if err := jsonfile.DecodeStrict(filepath.Join(root, "cmd/validate/testdata/media_pending.json"), &value); err != nil {
		return value, err
	}
	if value.Finding == "" || value.Owner == "" || value.Reason == "" {
		return value, errors.New("media pending: an addition names its finding, owner and reason")
	}
	return value, nil
}

// pends reports whether the capability is a pending addition.
func (p mediaPending) pends(model artifact.ID, task recipe.Task, definition artifact.ID) bool {
	return slices.Contains(p.Cells, mediaPendingCell{Model: model, Task: task, Recipe: definition})
}

// censusModels removes the pending capabilities from census entries.
func (p mediaPending) censusModels(models []censusModel) []censusModel {
	result := make([]censusModel, 0, len(models))
	for _, model := range models {
		model.Capabilities = slices.DeleteFunc(slices.Clone(model.Capabilities), func(capability censusCapability) bool {
			return p.pends(model.Model, capability.Task, capability.Recipe)
		})
		result = append(result, model)
	}
	return result
}

// catalogEntries removes the pending capabilities from catalog entries and
// drops an entry left with none.
func (p mediaPending) catalogEntries(entries []discovery.CatalogEntry) []discovery.CatalogEntry {
	result := make([]discovery.CatalogEntry, 0, len(entries))
	for _, entry := range entries {
		entry.Capabilities = slices.DeleteFunc(slices.Clone(entry.Capabilities), func(capability discovery.Capability) bool {
			return p.pends(entry.Model, capability.Task, capability.Recipe)
		})
		if len(entry.Capabilities) > 0 {
			result = append(result, entry)
		}
	}
	return result
}

// TestMediaPendingAdditions holds each pending addition to what the store
// holds and the inventory does not: a cell the live catalog still serves as
// listed and no frozen cell names, an inactive recipe in the listed state
// that the frozen inventory does not list. An addition its owner has frozen,
// or the store has moved, is refused, so the list ends with the gap.
func TestMediaPendingAdditions(t *testing.T) {
	t.Parallel()
	testskip.Short(t, "pending media additions are checked against the private store")
	root := testutil.RepoRoot(t)
	pending, err := loadMediaPending(root)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := loadFrozenMediaInventory(root)
	if err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	live, truncated, err := discovery.RegisteredCatalog(t.Context(), store, catalogLimit, discovery.LoadMemo(t.Context(), store))
	if err != nil || truncated {
		t.Fatalf("live registered catalog: truncated=%t: %v", truncated, err)
	}
	for _, cell := range pending.Cells {
		if !slices.Contains(inventory.Tasks, cell.Task) {
			t.Errorf("pending %s/%s is not a media task", cell.Model, cell.Task)
		}
		if slices.ContainsFunc(inventory.Cells, func(frozen imageVideoInventoryCell) bool {
			return frozen.Model == cell.Model && frozen.Task == cell.Task && frozen.Recipe == cell.Recipe
		}) {
			t.Errorf("pending %s/%s is frozen now; remove it from media_pending.json", cell.Model, cell.Task)
		}
		if !slices.ContainsFunc(live, func(entry discovery.CatalogEntry) bool {
			return entry.Model == cell.Model && slices.ContainsFunc(entry.Capabilities, func(capability discovery.Capability) bool {
				return capability.Task == cell.Task && capability.Recipe == cell.Recipe
			})
		}) {
			t.Errorf("pending %s/%s is no longer served as listed; remove or correct it", cell.Model, cell.Task)
		}
	}
	for _, inactive := range pending.Inactive {
		if slices.ContainsFunc(inventory.Inactive, func(frozen imageVideoInactive) bool { return frozen.Recipe == inactive.Recipe }) {
			t.Errorf("pending inactive %s is frozen now; remove it from media_pending.json", inactive.Recipe)
		}
		status, found, err := modelrecipe.Status(t.Context(), store, inactive.Recipe)
		if err != nil || !found || status != inactive.Status || status == recipe.StatusActive {
			t.Errorf("pending inactive %s: status %q found=%t, listed %q: %v", inactive.Recipe, status, found, inactive.Status, err)
		}
	}
}
