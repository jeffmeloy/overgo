package automationcheck

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestAgentHarnessBoundaryOwnership(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"internal/recipe/recipe.go":       "package recipe\ntype Task string\n",
		"internal/agenttool/tool.go":      "package agenttool\ntype Manual string\n",
		"internal/agentloop/loop.go":      "package agentloop\nimport (\n\"overgo/internal/recipe\"\n\"overgo/internal/agenttool\"\n)\nvar _, _ = recipe.Task(\"\"), agenttool.Manual(\"\")\n",
		"internal/agentloop/loop_test.go": "package agentloop\n",
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
	snapshot, err := repoanalysis.LoadGo(root, paths)
	if err != nil {
		t.Fatal(err)
	}
	coverage, err := AgentHarnessBoundaryCoverage(snapshot, []string{"internal/recipe/recipe.go"})
	if err != nil {
		t.Fatal(err)
	}
	if len(coverage.Changed) != 1 || len(coverage.Boundaries) != 1 || coverage.Boundaries[0] != "internal/agentloop" || len(coverage.Missing) != 0 {
		t.Fatalf("coverage = %+v", coverage)
	}
	if err := RequireAgentHarnessBoundaries(coverage, []string{"overgo/internal/recipe", "overgo/internal/agentloop"}); err != nil {
		t.Fatal(err)
	}
	if err := RequireAgentHarnessBoundaries(coverage, []string{"overgo/internal/recipe"}); err == nil {
		t.Fatal("missing assembled boundary passed selection")
	}
}
