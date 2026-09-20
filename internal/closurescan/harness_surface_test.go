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

// TestHarnessSurfaceReport holds the report to every metric once, in one order,
// with its baseline and its measure whether it moved or not, and holds the
// regressions to exactly the metrics that grew.
func TestHarnessSurfaceReport(t *testing.T) {
	base := HarnessSurface{ProductionFiles: 9, ProductionNodes: 100, MaxFileNodes: 50}
	moved := base
	moved.ProductionNodes, moved.MaxFileNodes = 90, 60
	report := HarnessSurfaceReport(base, moved)
	names := make([]string, 0, len(report))
	for _, metric := range report {
		names = append(names, metric.Metric)
	}
	if unique := slices.Compact(slices.Sorted(slices.Values(names))); len(unique) != 7 || names[1] != "production_nodes" || report[1].Base != 100 || report[1].Value != 90 {
		t.Fatalf("report = %+v", report)
	}
	grown := AgentHarnessSurfaceRegressions(base, moved)
	if len(grown) != 1 || grown[0].Metric != "max_file_nodes" || grown[0].Base != 50 || grown[0].Value != 60 {
		t.Fatalf("regressions = %+v, want only the metric that grew", grown)
	}
	if len(AgentHarnessSurfaceRegressions(base, base)) != 0 {
		t.Fatal("an unmoved surface regressed")
	}
}
