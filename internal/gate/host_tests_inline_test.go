package gate

import (
	"slices"
	"testing"

	"overgo/internal/testutil"
)

// TestHostTestsRunBeforeTheCommit holds the device lease to packages that
// can execute a device command: a source-reading package that runs git by
// name executes nothing of the repository's, so its tests run before the
// commit with the host packages; one that runs a program it computes may
// run any repository command, a device worker among them, and stays under
// the lease.
func TestHostTestsRunBeforeTheCommit(t *testing.T) {
	t.Parallel()
	g := scopeCompilerFixture(t)
	for name, content := range map[string]string{
		"internal/cuda/device/device.go": "package device\n",
		"cmd/cuda-smoke/main.go":         "package main\nimport _ \"overgo/internal/cuda/device\"\nfunc main() {}\n",
		"internal/gitreader/reader.go": "package gitreader\nimport (\"go/parser\"; \"os\"; \"os/exec\")\nvar _ = parser.ParseFile\n" +
			"func Read(path string) ([]byte, error) { return os.ReadFile(path) }\n" +
			"func Listed() ([]byte, error) { return exec.Command(\"git\", \"ls-files\").Output() }\n",
		"internal/dynrunner/runner.go": "package dynrunner\nimport (\"go/parser\"; \"os\"; \"os/exec\")\nvar _ = parser.ParseFile\n" +
			"func Read(path string) ([]byte, error) { return os.ReadFile(path) }\n" +
			"func Run() error { program := os.Getenv(\"OVERGO_TOOL\"); return exec.Command(program).Run() }\n",
	} {
		testutil.WriteTextFile(t, g.repo, name, content)
	}
	runGitFixture(t, g.repo, "add", ".")
	g.packageGraph = nil
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	devices, err := graph.devicePackages([]string{"overgo/internal/gitreader", "overgo/internal/dynrunner"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(devices, []string{"overgo/internal/dynrunner"}) {
		t.Fatalf("device packages = %v, want the computed-program runner alone", devices)
	}
}
