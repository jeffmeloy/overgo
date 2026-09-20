package gate

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/plan"
	"overgo/internal/repoanalysis"
)

// TestDerivedShipPaths binds the ship set of a run that names no -paths:
// every dirty path once and sorted, both sides of a rename, the plan alone
// when nothing is dirty, and a refusal naming any top-level entry HEAD does
// not track -- a stray directory or a new root file must be named to ship.
func TestDerivedShipPaths(t *testing.T) {
	t.Parallel()
	tracked := []string{"cmd", "docs", "go.mod", "internal"}
	paths, err := deriveShipPaths([]repoanalysis.DirtyPath{
		{Path: "internal/gate/run.go"},
		{Path: "cmd/gate/main.go"},
		{Path: "docs/new_receipt.json"},
		{Path: "internal/plan/b.go", OriginalPath: "internal/plan/a.go"},
		{Path: "go.mod"},
	}, tracked)
	want := []string{"cmd/gate/main.go", "docs/new_receipt.json", "go.mod", "internal/gate/run.go", "internal/plan/a.go", "internal/plan/b.go"}
	if err != nil || !slices.Equal(paths, want) {
		t.Fatalf("derived ship set = (%v, %v), want %v", paths, err, want)
	}
	if paths, err := deriveShipPaths(nil, tracked); err != nil || !slices.Equal(paths, []string{plan.Path}) {
		t.Fatalf("clean tree ship set = (%v, %v)", paths, err)
	}
	for _, stray := range []string{"overgodb-store.pre-compact/blobs/evidence/aa/aa11", "notes.md"} {
		_, err := deriveShipPaths([]repoanalysis.DirtyPath{{Path: "internal/gate/run.go"}, {Path: stray}}, tracked)
		top, _, _ := strings.Cut(stray, "/")
		if err == nil || !strings.Contains(err.Error(), top) {
			t.Fatalf("stray %s = %v, want a refusal naming %s", stray, err, top)
		}
	}
}
