package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestNamedDocumentReach pins the binding of a changed document: a document
// some package names in source binds to its namers and their importers
// alone, a document nobody names still binds to every unnamed reader. On
// the live graph the regenerated documents every source commit carries then
// add no package to a gate-only working set.
func TestNamedDocumentReach(t *testing.T) {
	t.Run("fixture", namedDocumentFixture)
	root := liveGateContext(t).repo
	complete := map[string][]string{}
	scope := func(paths ...string) []string {
		g := liveGateContext(t, paths...)
		derived, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		selected := derived.selected()
		slices.Sort(selected)
		complete[strings.Join(paths, ",")] = derived.dependent
		t.Logf("paths=%v direct=%d uncertain=%d dependent=%d excluded=%d", paths, len(derived.direct), len(derived.uncertain), len(derived.dependent), derived.excluded)
		return selected
	}
	const manifest = "docs/api_manifest.json"
	g := liveGateContext(t, manifest)
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	readers := graph.namedReaders(manifest)
	if len(readers) == 0 {
		t.Fatalf("%s has no named reader", manifest)
	}
	document := scope(manifest)
	// Every package the document selects names it or compiles a namer.
	for _, selected := range document {
		if slices.ContainsFunc(graph.byID[selected], func(index int) bool { return namesPath(graph.nodes[index].declaredFiles, manifest) }) {
			continue
		}
		compiled, err := graph.inputNodes(selected, true)
		if err != nil {
			t.Fatal(err)
		}
		reaches := false
		for index := range compiled {
			if dir, err := filepath.Rel(root, graph.nodes[index].Dir); err == nil && slices.Contains(readers, filepath.ToSlash(dir)) {
				reaches = true
				break
			}
		}
		if !reaches {
			t.Errorf("%s selected by %s without naming it or compiling a namer %v", selected, manifest, readers)
		}
	}
	// Measured 2026-09-14: the manifest alone selected 231 of 256 packages
	// before the rule and 17 after; the ratchet holds a fifth.
	total := 0
	for _, node := range graph.nodes {
		if len(node.Match) != 0 && node.ForTest == "" {
			total++
		}
	}
	if len(document) >= total/5 {
		t.Fatalf("%s selects %d packages of %d; the named-document binding is not in effect", manifest, len(document), total)
	}
	gateOnly := scope("internal/gate/preflight.go")
	combined := scope(manifest, "docs/modern_go_baseline.json", "docs/modern_go_census.json", "internal/gate/preflight.go")
	for _, selected := range combined {
		if !slices.Contains(gateOnly, selected) && !slices.Contains(document, selected) {
			t.Errorf("%s selected by the combined working set but by neither input alone", selected)
		}
	}
	if len(combined) > len(gateOnly)+len(document) {
		t.Fatalf("combined working set selected %d packages, more than %d + %d", len(combined), len(gateOnly), len(document))
	}
	// A document change runs its readers and their importers short; only
	// the packages that compile the gate file take the complete group.
	if got, want := complete[strings.Join([]string{manifest, "docs/modern_go_baseline.json", "docs/modern_go_census.json", "internal/gate/preflight.go"}, ",")], complete["internal/gate/preflight.go"]; !slices.Equal(got, want) {
		t.Fatalf("the regenerated documents changed the complete group: %v, gate-only %v", got, want)
	}
}

// namedDocumentFixture: the reader names docs/config.txt and other reads a
// path from the environment. The named document selects the reader and its
// importer and changes the reader identity, not the unnamed reader; a
// document nobody names selects the unnamed reader and not the namer.
func namedDocumentFixture(t *testing.T) {
	g := runtimeReaderFixture(t)
	write := func(name, content string) {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("docs/other.txt", "1\n")
	write("internal/other/other.go", "package other\nimport \"os\"\nfunc Value() ([]byte, error) { return os.ReadFile(os.Getenv(\"INPUT\")) }\n")
	write("internal/other/other_test.go", "package other\nimport \"testing\"\nfunc TestValue(t *testing.T) { Value() }\n")
	runGitFixture(t, g.repo, "add", ".")
	scope := func(paths ...string) []string {
		g.paths, g.packageGraph = paths, nil
		derived, err := g.deriveTestScope()
		if err != nil {
			t.Fatal(err)
		}
		return derived.selected()
	}
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	before, err := graph.identity("overgo/internal/reader")
	if err != nil {
		t.Fatal(err)
	}
	named := scope("docs/config.txt")
	for _, required := range []string{"overgo/internal/reader", "overgo/internal/readerclient"} {
		if !slices.Contains(named, required) {
			t.Fatalf("named document dropped %s: %v", required, named)
		}
	}
	if slices.Contains(named, "overgo/internal/other") {
		t.Fatalf("named document selected the unnamed reader: %v", named)
	}
	if err := os.WriteFile(filepath.Join(g.repo, "docs", "config.txt"), []byte("2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g.packageGraph = nil
	graph, err = g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	after, err := graph.identity("overgo/internal/reader")
	if err != nil || before == after {
		t.Fatalf("named document change retained the reader identity: %v", err)
	}
	unnamed := scope("docs/other.txt")
	if !slices.Contains(unnamed, "overgo/internal/other") {
		t.Fatalf("unnamed document dropped the unnamed reader: %v", unnamed)
	}
	if slices.Contains(unnamed, "overgo/internal/reader") || slices.Contains(unnamed, "overgo/internal/readerclient") {
		t.Fatalf("unnamed document selected the namer of another document: %v", unnamed)
	}
}
