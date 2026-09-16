package gate

import (
	"path/filepath"
	"slices"
	"testing"
)

// Source provenance must retain producers without importing every analyzer's reads.
func TestAudioProducerInputBoundary(t *testing.T) {
	liveRepositoryFixture(t).use(t, func(g *gateContext) {
		graph, err := g.inputGraph()
		if err != nil {
			t.Fatal(err)
		}
		const target = "overgo/internal/audioparity"
		inputs, err := graph.inputFiles(target)
		if err != nil {
			t.Fatal(err)
		}
		for path, want := range map[string]bool{
			"internal/gate/preparation.go":                   false,
			"docs/api_manifest.json":                         false,
			"docs/modern_go_baseline.json":                   false,
			"docs/modern_go_census.json":                     false,
			"internal/audioparity/cpu_baseline_test.go":      true,
			"internal/evaluation/transcription_resources.go": true,
			"cmd/evaluate/main.go":                           true,
			"cmd/recipe/main.go":                             true,
		} {
			if got := inputs[filepath.Join(g.sourceRoot(), filepath.FromSlash(path))]; got != want {
				t.Errorf("%s input=%t, want %t", path, got, want)
			}
			g.paths = []string{path}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			if got := slices.Contains(scope.selected(), target); got != want {
				t.Errorf("%s selects audio=%t, want %t", path, got, want)
			}
		}
	})
}
