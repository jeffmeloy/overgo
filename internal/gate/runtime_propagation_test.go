package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

// runtimeReaderFixture: a library that reads a repository file at run time
// and has no test of its own, a caller whose test runs that read, and a
// package touching neither. The compiler fixture keeps its own package
// set, whose expectations enumerate it.
func runtimeReaderFixture(t *testing.T) *gateContext {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":                               "module overgo\n\ngo 1.25\n",
		"docs/config.txt":                      "1\n",
		"internal/reader/reader.go":            "package reader\nimport (\"os\"; \"strconv\"; \"strings\")\nfunc Value() int {\n\tdata, err := os.ReadFile(\"../../docs/config.txt\")\n\tif err != nil {\n\t\treturn -1\n\t}\n\tvalue, _ := strconv.Atoi(strings.TrimSpace(string(data)))\n\treturn value\n}\n",
		"internal/readerclient/client.go":      "package readerclient\nimport \"overgo/internal/reader\"\nfunc Value() int { return reader.Value() }\n",
		"internal/readerclient/client_test.go": "package readerclient\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value() != 1 { t.Fatal(Value()) } }\n",
		"internal/unrelated/unrelated.go":      "package unrelated\nconst Value = 1\n",
		"internal/unrelated/unrelated_test.go": "package unrelated\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value != 1 { t.Fatal(Value) } }\n",
		"internal/isolated/isolated.go":        "package isolated\nimport \"overgo/internal/unrelated\"\nfunc Value() int { return unrelated.Value }\n",
		"internal/isolated/isolated_test.go":   "package isolated\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value() != 1 { t.Fatal(Value()) } }\n",
	}
	var sources []string
	for name, content := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		if strings.HasSuffix(name, ".go") {
			sources = append(sources, name)
		}
	}
	runGitFixture(t, root, "init", "-q")
	runGitFixture(t, root, "add", ".")
	snapshot, err := repoanalysis.LoadGo(root, sources)
	if err != nil {
		t.Fatal(err)
	}
	return &gateContext{repo: root, source: &snapshot}
}

// TestRuntimeInputsPropagateThroughCallers pins the owner's counterexample:
// a change to a file a library reads at run time selects the library's
// caller beside the reader in the short group and changes the caller's
// package input identity, while a package importing nothing that reads the
// file is excluded and keeps its identity.
func TestRuntimeInputsPropagateThroughCallers(t *testing.T) {
	t.Parallel()
	g := runtimeReaderFixture(t)
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	before := map[string]string{}
	for _, pkg := range []string{"overgo/internal/readerclient", "overgo/internal/isolated"} {
		identity, err := graph.identity(pkg)
		if err != nil {
			t.Fatal(err)
		}
		before[pkg] = identity.String()
	}
	g.paths = []string{"docs/config.txt"}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	// The reader itself is a direct owner of the file it names; its caller
	// inherits the read and runs in the same short group.
	if !slices.Contains(scope.uncertain, "overgo/internal/readerclient") || !slices.Contains(scope.direct, "overgo/internal/reader") {
		t.Fatalf("the library's caller is not selected beside the reader: direct=%v uncertain=%v dependent=%v", scope.direct, scope.uncertain, scope.dependent)
	}
	if slices.Contains(scope.selected(), "overgo/internal/isolated") || scope.excluded == 0 {
		t.Fatalf("a package reading nothing was selected: direct=%v uncertain=%v dependent=%v excluded=%d", scope.direct, scope.uncertain, scope.dependent, scope.excluded)
	}
	if err := os.WriteFile(filepath.Join(g.repo, "docs", "config.txt"), []byte("2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	graph, err = g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	for pkg, want := range map[string]bool{"overgo/internal/readerclient": true, "overgo/internal/isolated": false} {
		identity, err := graph.identity(pkg)
		if err != nil {
			t.Fatal(err)
		}
		if changed := identity.String() != before[pkg]; changed != want {
			t.Fatalf("%s identity changed=%t after the library's runtime input changed, want %t", pkg, changed, want)
		}
	}
}
