package gate

import (
	"slices"
	"strings"
	"testing"
)

// TestSurfaceMovePreflightNamesAnInferenceSurfaceMove holds the surface check
// to advising, from the gate's own ship set, that a landing moves the
// inference surface. Validation evidence is keyed to that surface: a source
// added to, edited in or removed from a package of the closure expires the
// long-form guard record of every model, and a test file, a document or a
// package outside the closure expires none. The warning states how many
// sources move while naming a bounded few, and a landing that moves nothing
// is not warned. The closure is the live one.
func TestSurfaceMovePreflightNamesAnInferenceSurfaceMove(t *testing.T) {
	t.Parallel()
	warnings := func(paths []string) []string {
		g := &gateContext{repo: "../..", paths: paths}
		_, _ = g.stepSurface() // the media and pin verdicts are other tests' subject
		var found []string
		for _, note := range g.audit {
			if note.kind == noteWarning {
				found = append(found, note.text)
			}
		}
		return found
	}
	moving := warnings([]string{
		"internal/modelartifact/registration.go", "internal/modelartifact/registration_assemble.go",
		"internal/modelartifact/registration_test.go", "cmd/recipe/spec.go", "docs/plan.json",
		"internal/longform/old.go", "internal/inference/moved.go",
	})
	if len(moving) != 1 {
		t.Fatalf("surface warnings = %q, want one", moving)
	}
	for _, part := range []string{"moves the inference surface (3 source(s):", "internal/modelartifact/registration_assemble.go", "internal/inference/moved.go", "guard record"} {
		if !strings.Contains(moving[0], part) {
			t.Errorf("warning lacks %q: %s", part, moving[0])
		}
	}
	many := make([]string, surfaceMoveShown+3)
	for index := range many {
		many[index] = "internal/inference/file" + string(rune('a'+index)) + ".go"
	}
	if bounded := warnings(many); len(bounded) != 1 || !strings.Contains(bounded[0], "(9 source(s):") || strings.Contains(bounded[0], many[surfaceMoveShown]) {
		t.Errorf("a long move list was not bounded: %q", bounded)
	}
	if quiet := warnings([]string{"docs/plan.json", "internal/modelartifact/registration_test.go"}); !slices.Equal(quiet, nil) {
		t.Errorf("a landing that moves nothing warned: %q", quiet)
	}
}
