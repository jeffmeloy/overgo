package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestRuntimeOpaqueCallerSourceInput pins the supported boundary of an
// unnamed read whose path arrives from the environment: it reaches the
// repository's data files, so a data change selects and invalidates the
// reader's caller and the seeded regression is detected; it does not reach
// Go source, which only a Go-parsing or program-running reach observes, so
// a source change outside the compiled closure selects nothing.
func TestRuntimeOpaqueCallerSourceInput(t *testing.T) {
	g := runtimeReaderFixture(t)
	source := `package reader
import ("os";"strings")
func Value() int { b,e:=os.ReadFile(os.Getenv("INPUT"));if e==nil&&strings.Contains(string(b),"1"){return 1};return -1 }
`
	if e := os.WriteFile(filepath.Join(g.repo, "internal", "reader", "reader.go"), []byte(source), 0644); e != nil {
		t.Fatal(e)
	}
	input := filepath.Join(g.repo, "docs", "config.txt")
	t.Setenv("INPUT", input)
	g.paths = []string{"docs/config.txt"}
	s, e := g.deriveTestScope()
	if e != nil {
		t.Fatal(e)
	}
	if !slices.Contains(s.selected(), "overgo/internal/readerclient") {
		t.Fatalf("opaque reader caller omitted for a data change: direct=%v dependent=%v", s.direct, s.dependent)
	}
	if output, err := command(g.repo, "go", "test", "./internal/readerclient", "-count=1"); err != nil {
		t.Fatalf("baseline caller: %v\n%s", err, output)
	}
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	before, err := graph.identity("overgo/internal/readerclient")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(input, []byte("2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	after, err := graph.identity("overgo/internal/readerclient")
	if err != nil || after == before {
		t.Fatalf("runtime data mutation retained caller identity: %v", err)
	}
	if output, err := command(g.repo, "go", "test", "./internal/readerclient", "-count=1"); err == nil {
		t.Fatalf("seeded runtime regression escaped the selected caller: %s", output)
	}

	// Boundary: the same reader is not selected by a Go source change it
	// neither compiles nor parses; the classifier records the reach as data.
	g.paths = []string{"internal/unrelated/unrelated.go"}
	g.packageGraph = nil
	s, e = g.deriveTestScope()
	if e != nil {
		t.Fatal(e)
	}
	if slices.Contains(s.dependent, "overgo/internal/readerclient") || slices.Contains(s.dependent, "overgo/internal/reader") {
		t.Fatalf("data reader selected by a source change outside its closure: direct=%v dependent=%v", s.direct, s.dependent)
	}
	if !slices.Contains(s.direct, "overgo/internal/unrelated") {
		t.Fatalf("source owner lost: %v", s.direct)
	}
}
