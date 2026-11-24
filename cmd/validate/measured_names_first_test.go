package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// TestAnUnknownMeasuredNameIsRefusedBeforeAnyCellIsBound holds measured-names-first (SQA-478): a full run whose
// measured list names a model no registered model answers to is refused with that name before any cell is bound, in a
// checkout whose projection suites are absent, where binding alone fails on the suite it cannot read; a run that reads
// no list (-inventory, -model, -trained) binds as before. The full measurement's preflight at 3fe41bfa0 refused a
// retired Gemma name only after binding every cell, about two minutes in, and an absent suite would have hidden it.
func TestAnUnknownMeasuredNameIsRefusedBeforeAnyCellIsBound(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	list := filepath.Join(root, filepath.FromSlash(measuredModelsFile))
	testcheck.NoError(t, os.MkdirAll(filepath.Dir(list), 0o755))
	testcheck.NoError(t, os.WriteFile(list, []byte(`{"models": [{"name": "a-retired-model", "path": "nothing registered"}]}`), 0o600))
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	if _, _, err := chosenThenBound(t.Context(), root, store, nil, "", false); err == nil {
		t.Fatal("binding in a checkout with no projection suite passed; want it refused, so the order below is what is held")
	}
	_, _, err := chosenThenBound(t.Context(), root, store, nil, "", true)
	if err == nil || !strings.Contains(err.Error(), `"a-retired-model", which no registered model answers to`) {
		t.Fatalf("a full run whose list names an unknown model was refused with %v; want the name, before any cell is bound", err)
	}
}
