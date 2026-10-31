//go:build modeltest && integration

package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/hfconvert"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestEveryViewedTextModelServesFromItsDirectory holds the live store to serving every text model a view serves from
// its own files: no model directory of an architecture the converter's view serves (hfconvert.Viewed) has a GGUF
// conversion still activated for inference beside it. A conversion activated for training alone stays, as its training
// recipe reads it. Qwen3.5-4B moved on 2026-10-30; Qwen2.5-0.5B, Carbon-500M and MiniCPM5-1B on 2026-10-31.
func TestEveryViewedTextModelServesFromItsDirectory(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	registered, err := registeredCells(t.Context(), store)
	if err != nil {
		t.Fatal(err)
	}
	converted, err := convertedSources(t.Context(), store)
	if err != nil {
		t.Fatal(err)
	}
	directories, err := modelDirectories(roots.ModelRoots())
	if err != nil {
		t.Fatal(err)
	}
	viewed := 0
	for _, entry := range buildInventory(directories, registered, converted, roots.Store) {
		if !hfconvert.Viewed(entry.Directory) {
			continue
		}
		viewed++
		for _, served := range entry.Models {
			conversion := strings.EqualFold(filepath.Ext(served.Location), ".gguf")
			if conversion && slices.ContainsFunc(served.Cells, func(cell ModelValidation) bool { return cell.Kind == recipe.TaskInference }) {
				t.Errorf("%s serves inference from its conversion %s, not from its own files: onboard it (validate -onboard)", entry.Directory, served.Location)
			}
		}
	}
	if viewed == 0 {
		t.Fatal("the model roots hold no directory a view serves")
	}
	t.Logf("%d model directories a view serves", viewed)
}
