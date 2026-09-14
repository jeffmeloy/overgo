package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestRuntimeDataDependencyDoesNotReserveDevice(t *testing.T) {
	g := scopeCompilerFixture(t)
	for name, content := range map[string]string{
		"internal/cuda/executor/executor.go":  "package executor\nconst Value = 1\n",
		"internal/datareader/reader.go":       "package datareader\nimport \"os\"\nfunc Read(path string) ([]byte,error) { return os.ReadFile(path) }\n",
		"internal/launcher/launcher.go":       "package launcher\nimport \"os/exec\"\nfunc Run(path string) error { return exec.Command(path).Run() }\n",
		"internal/deviceconsumer/consumer.go": "package deviceconsumer\nimport \"overgo/internal/cuda/executor\"\nconst Value = executor.Value\n",
	} {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGitFixture(t, g.repo, "add", ".")
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	deviceOwners, err := graph.dependentDirectories("internal/cuda")
	if err != nil {
		t.Fatal(err)
	}
	for _, owner := range []string{"internal/cuda/executor", "internal/deviceconsumer"} {
		if !slices.Contains(deviceOwners, owner) {
			t.Errorf("device execution owner omitted: %s", owner)
		}
	}
	// A launcher runs the program its caller hands in: the device
	// requirement belongs to the caller that names a device program.
	for _, other := range []string{"internal/datareader", "internal/launcher"} {
		if slices.Contains(deviceOwners, other) {
			t.Errorf("%s acquired execution's device requirement without naming a device program", other)
		}
	}
	// An unnamed read reaches the repository's data files, not its Go
	// sources: a source change outside the reader's compiled closure leaves
	// the reader unselected with its identity intact.
	const reader = "overgo/internal/datareader"
	before, err := graph.identity(reader)
	if err != nil {
		t.Fatal(err)
	}
	path := "internal/cuda/executor/executor.go"
	if err := os.WriteFile(filepath.Join(g.repo, filepath.FromSlash(path)), []byte("package executor\nconst Value = 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g.paths = []string{path}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(scope.selected(), reader) {
		t.Error("data reader selected by a source change outside its closure")
	}
	after, err := graph.identity(reader)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Error("source change outside the closure altered the data reader identity")
	}
}
