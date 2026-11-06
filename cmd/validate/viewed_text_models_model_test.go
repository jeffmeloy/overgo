//go:build modeltest && integration

package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/hfconvert"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestEveryViewedTextModelServesFromItsDirectory holds the live store to serving every text model a view serves from
// its own files: no model directory of an architecture the converter's view serves (hfconvert.Viewed) has a GGUF
// conversion still activated for inference beside it. A conversion activated for training alone stays, as its training
// recipe reads it. Qwen3.5-4B moved on 2026-10-30; Qwen2.5-0.5B, Carbon-500M and MiniCPM5-1B on 2026-10-31.
func TestEveryViewedTextModelServesFromItsDirectory(t *testing.T) {
	t.Parallel()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := overgodbtest.OpenReadOnly(t, roots.Store)
	registered := testcheck.Must(registeredCells(t.Context(), store))(t)
	converted := testcheck.Must(convertedSources(t.Context(), store))(t)
	directories := testcheck.Must(modelDirectories(roots.ModelRoots()))(t)
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
