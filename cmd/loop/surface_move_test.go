package main

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/inferencesurface"
)

// TestLandingNamesAnInferenceSurfaceMove holds the landing to saying, before
// the gate, that its ship set moves the inference surface. Validation evidence
// is keyed to that surface: a source added to, edited in or removed from a
// package of the closure expires the long-form guard record of every model,
// and a test file, a document or a package outside the closure expires none.
// The ship set is the dirty tree, a rename counts by both of its paths, and
// the warning states how many sources move while naming a bounded few. The
// closure is the live one, so a package that leaves or joins it is followed.
func TestLandingNamesAnInferenceSurfaceMove(t *testing.T) {
	t.Parallel()
	porcelain := " M internal/modelartifact/registration.go\n" +
		"D  internal/modelartifact/registration_assemble.go\n" +
		"?? internal/modelartifact/registration_test.go\n" +
		" M cmd/recipe/spec.go\n" +
		" M docs/plan.json\n" +
		"R  internal/longform/old.go -> internal/inference/moved.go\n" +
		" M kernels/cuda/attention.cu\n"
	paths := shipSetPaths(porcelain)
	want := []string{
		"internal/modelartifact/registration.go", "internal/modelartifact/registration_assemble.go",
		"internal/modelartifact/registration_test.go", "cmd/recipe/spec.go", "docs/plan.json",
		"internal/longform/old.go", "internal/inference/moved.go", "kernels/cuda/attention.cu",
	}
	if !slices.Equal(paths, want) {
		t.Fatalf("ship set paths = %v\nwant %v", paths, want)
	}
	moves, err := inferencesurface.Moves(t.Context(), "../..", paths)
	if err != nil {
		t.Fatal(err)
	}
	moved := []string{
		"internal/inference/moved.go", "internal/modelartifact/registration.go",
		"internal/modelartifact/registration_assemble.go", "kernels/cuda/attention.cu",
	}
	if !slices.Equal(moves, moved) {
		t.Fatalf("surface moves = %v\nwant %v", moves, moved)
	}
	warning := surfaceMoveWarning(moves)
	for _, part := range []string{"advisory: warning:", "moves the inference surface (4 source(s):", "internal/modelartifact/registration_assemble.go", "guard record"} {
		if !strings.Contains(warning, part) {
			t.Errorf("warning lacks %q: %s", part, warning)
		}
	}
	many := make([]string, surfaceMoveShown+3)
	for index := range many {
		many[index] = "internal/inference/file" + string(rune('a'+index)) + ".go"
	}
	if bounded := surfaceMoveWarning(many); !strings.Contains(bounded, "(9 source(s):") || strings.Contains(bounded, many[surfaceMoveShown]) {
		t.Errorf("a long move list was not bounded: %s", bounded)
	}
	if quiet := surfaceMoveWarning(nil); quiet != "" {
		t.Errorf("a landing that moves nothing warned: %s", quiet)
	}
}
