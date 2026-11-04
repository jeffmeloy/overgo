package landing

import (
	"strings"
	"testing"
)

// TestTheTestStageRunsEveryPackage changes one package of a module whose other package has a failing test, and holds
// the test stage to refusing it: the stage runs every package and native go test chooses what to run again from its
// own cache, so no selection narrower than the module lets a broken package the change never touched pass.
func TestTheTestStageRunsEveryPackage(t *testing.T) {
	t.Parallel()
	fixture := newSliceFixture(t, map[string]string{
		"p/p.go":      "// Package p is a fixture.\npackage p\n",
		"q/q.go":      "// Package q is a fixture.\npackage q\n",
		"q/q_test.go": "package q\n\nimport \"testing\"\n\nfunc TestUntouched(t *testing.T) { t.Fatal(\"broken where the change never went\") }\n",
	})
	fixture.write("p/p.go", "// Package p is a fixture, changed.\npackage p\n")
	l := &landing{Options: Options{Repo: fixture.repo}, details: map[string]string{}}
	if err := l.test(t.Context()); err == nil || !strings.Contains(err.Error(), "fixture/q") {
		t.Fatalf("the test stage over a change to p ended as %v, want q's failing test to refuse it", err)
	}
}
