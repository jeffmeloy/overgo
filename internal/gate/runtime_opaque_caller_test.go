package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestRuntimeOpaqueCallerSourceInput(t *testing.T) {
	g := runtimeReaderFixture(t)
	source := `package reader
import ("os";"strings")
func Value() int { b,e:=os.ReadFile(os.Getenv("INPUT"));if e==nil&&strings.Contains(string(b),"Value = 1"){return 1};return -1 }
`
	if e := os.WriteFile(filepath.Join(g.repo, "internal", "reader", "reader.go"), []byte(source), 0644); e != nil {
		t.Fatal(e)
	}
	g.paths = []string{"internal/unrelated/unrelated.go"}
	s, e := g.deriveTestScope()
	if e != nil {
		t.Fatal(e)
	}
	if !slices.Contains(s.dependent, "overgo/internal/readerclient") {
		t.Fatalf("opaque reader caller omitted: direct=%v dependent=%v", s.direct, s.dependent)
	}
	input := filepath.Join(g.repo, "internal", "unrelated", "unrelated.go")
	t.Setenv("INPUT", input)
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
	if err := os.WriteFile(input, []byte("package unrelated\nconst Value = 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	after, err := graph.identity("overgo/internal/readerclient")
	if err != nil || after == before {
		t.Fatalf("runtime source mutation retained caller identity: %v", err)
	}
	if output, err := command(g.repo, "go", "test", "./internal/readerclient", "-count=1"); err == nil {
		t.Fatalf("seeded runtime regression escaped the selected caller: %s", output)
	}
}
