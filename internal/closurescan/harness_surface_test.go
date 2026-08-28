package closurescan

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestAgentHarnessSurfaceRatchets(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"internal/recipe/state.go": `package recipe
import "sync"
type state struct { mu sync.Mutex; ready bool }
func use(v int) bool { return v > 7 }
`,
		"internal/agentloop/loop.go": `package agentloop
import "overgo/internal/recipe"
var _ = recipe.Task("")
`,
	}
	var paths []string
	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, name)
	}
	baseSnapshot, err := repoanalysis.LoadGo(root, paths)
	if err != nil {
		t.Fatal(err)
	}
	base, err := BuildAgentHarnessSurface(baseSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	if base.GuardedScalarFields != 1 || len(base.LayerViolations) != 0 {
		t.Fatalf("base surface = %+v", base)
	}
	candidateSnapshot, err := baseSnapshot.Overlay(map[string][]byte{
		"internal/recipe/state.go": []byte(`package recipe
import (
    "sync"
    "overgo/internal/agentloop"
)
type state struct { mu sync.Mutex; ready, closed bool }
func use(v int) bool { _ = agentloop.Session{}; return v > 7 }
`),
	})
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := BuildAgentHarnessSurface(candidateSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	regressions := AgentHarnessSurfaceRegressions(base, candidate)
	for _, metric := range []string{"guarded_scalar_fields", "layer_violation"} {
		if !slices.ContainsFunc(regressions, func(row HarnessSurfaceRegression) bool { return row.Metric == metric }) {
			t.Fatalf("regressions %v do not contain %s", regressions, metric)
		}
	}
	census, err := BuildCensus(candidateSnapshot)
	if err != nil {
		t.Fatal(err)
	}
	if census.Harness.GuardedScalarFields != candidate.GuardedScalarFields || len(census.Harness.LayerViolations) != 1 {
		t.Fatalf("census harness = %+v, want %+v", census.Harness, candidate)
	}
}
