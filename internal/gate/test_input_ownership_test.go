package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestImportedTestInputsDoNotInvalidateCaller(t *testing.T) {
	g := runtimeReaderFixture(t)
	write := func(name, source string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(g.repo, filepath.FromSlash(name)), []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/reader/reader.go", "package reader\nfunc Value() int { return 1 }\n")
	write("internal/reader/reader_test.go", "package reader\nimport (\"os\";\"strings\";\"testing\")\nfunc TestFixture(t *testing.T){ b,e:=os.ReadFile(os.Getenv(\"INPUT\")); if e!=nil||strings.TrimSpace(string(b))!=\"1\"{t.Fatalf(\"fixture: %s %v\",b,e)} }\n")
	t.Setenv("INPUT", filepath.Join(g.repo, "docs", "config.txt"))
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	callerBefore, err := graph.identity("overgo/internal/readerclient")
	if err != nil {
		t.Fatal(err)
	}
	readerBefore, err := graph.identity("overgo/internal/reader")
	if err != nil {
		t.Fatal(err)
	}
	if output, err := command(g.repo, "go", "test", "./internal/reader", "./internal/readerclient", "-count=1"); err != nil {
		t.Fatalf("baseline: %v\n%s", err, output)
	}
	write("docs/config.txt", "2\n")
	callerAfter, err := graph.identity("overgo/internal/readerclient")
	if err != nil || callerAfter != callerBefore {
		t.Fatalf("uncompiled dependency tests invalidated caller: %v", err)
	}
	readerAfter, err := graph.identity("overgo/internal/reader")
	if err != nil || readerAfter == readerBefore {
		t.Fatalf("own test fixture retained identity: %v", err)
	}
	if output, err := command(g.repo, "go", "test", "./internal/readerclient", "-count=1"); err != nil {
		t.Fatalf("unaffected caller: %v\n%s", err, output)
	}
	if output, err := command(g.repo, "go", "test", "./internal/reader", "-count=1"); err == nil {
		t.Fatalf("seeded fixture regression escaped: %s", output)
	}
	write("internal/unrelated/unrelated.go", "package unrelated\nconst Value = 2\n")
	callerAfter, err = graph.identity("overgo/internal/readerclient")
	if err != nil || callerAfter != callerBefore {
		t.Fatalf("dependency test runtime edge invalidated caller: %v", err)
	}
	write("internal/reader/reader_test.go", "package reader\nimport \"testing\"\nfunc TestFixture(t *testing.T){ t.Fatal(\"changed test\") }\n")
	callerAfter, err = graph.identity("overgo/internal/readerclient")
	if err != nil || callerAfter != callerBefore {
		t.Fatalf("uncompiled dependency source invalidated caller: %v", err)
	}
}

func TestTestCommandInputsBelongToOwningPackage(t *testing.T) {
	g := runtimeReaderFixture(t)
	write := func(name, source string) {
		t.Helper()
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/reader/reader.go", "package reader\nfunc Value() int { return 1 }\n")
	write("internal/reader/reader_test.go", "package reader\nimport (\"os/exec\";\"testing\")\nfunc TestCommand(t *testing.T){b,e:=exec.Command(\"go\",\"run\",\"../../cmd/probe\").CombinedOutput();if e!=nil||string(b)!=\"1\"{t.Fatalf(\"command: %s %v\",b,e)}}\n")
	write("cmd/probe/main.go", "package main\nimport \"fmt\"\nfunc main(){fmt.Print(\"1\")}\n")
	g.paths = []string{"cmd/probe/main.go"}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	selected := scope.selected()
	if !slices.Contains(selected, "overgo/internal/reader") || slices.Contains(selected, "overgo/internal/readerclient") {
		t.Fatalf("test command selected wrong owners: %v", selected)
	}
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	before, err := packageInputIdentities(graph, []string{"overgo/internal/reader", "overgo/internal/readerclient"})
	if err != nil {
		t.Fatal(err)
	}
	if output, err := command(g.repo, "go", "test", "./internal/reader", "./internal/readerclient", "-count=1"); err != nil {
		t.Fatalf("baseline: %v\n%s", err, output)
	}
	write("cmd/probe/main.go", "package main\nimport \"fmt\"\nfunc main(){fmt.Print(\"2\")}\n")
	after, err := packageInputIdentities(graph, []string{"overgo/internal/reader", "overgo/internal/readerclient"})
	if err != nil {
		t.Fatal(err)
	}
	if before["overgo/internal/reader"] == after["overgo/internal/reader"] || before["overgo/internal/readerclient"] != after["overgo/internal/readerclient"] {
		t.Fatal("test command identity escaped its owning test execution")
	}
	if output, err := command(g.repo, "go", "test", "./internal/readerclient", "-count=1"); err != nil {
		t.Fatalf("unaffected importer: %v\n%s", err, output)
	}
	if output, err := command(g.repo, "go", "test", "./internal/reader", "-count=1"); err == nil {
		t.Fatalf("seeded command regression escaped: %s", output)
	}
}

// TestRuntimeReaderLeavesUncompiledTestSource pins the data-reach boundary
// for an unnamed read: a reader that neither parses Go nor runs a program
// observes the repository's data files, so an uncompiled test source it is
// pointed at through the environment is outside its reach; the caller's
// identity holds and neither package is selected for that source change.
func TestRuntimeReaderLeavesUncompiledTestSource(t *testing.T) {
	g := runtimeReaderFixture(t)
	source := "package reader\nimport(\"os\";\"strings\")\nfunc Value()int{return read(os.Getenv(\"INPUT\"))}\nfunc read(path string)int{b,e:=os.ReadFile(path);if e==nil&&strings.Contains(string(b),\"Value = 1\"){return 1};return -1}\nfunc fixture(){path:=os.TempDir();_,_=os.ReadFile(path)}\n"
	if err := os.WriteFile(filepath.Join(g.repo, "internal", "reader", "reader.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(g.repo, "internal", "unrelated", "unrelated_test.go")
	if err := os.WriteFile(input, []byte("package unrelated\n// Value = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("INPUT", input)
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	before, err := graph.identity("overgo/internal/readerclient")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(input, []byte("package unrelated\n// Value = 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	after, err := graph.identity("overgo/internal/readerclient")
	if err != nil || before != after {
		t.Fatalf("uncompiled test source altered the data reader caller identity: %v", err)
	}
	g.paths = []string{"internal/unrelated/unrelated_test.go"}
	g.packageGraph = nil
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	for _, pkg := range []string{"overgo/internal/reader", "overgo/internal/readerclient"} {
		if slices.Contains(scope.selected(), pkg) {
			t.Fatalf("data reader %s selected by an uncompiled test source change: direct=%v dependent=%v", pkg, scope.direct, scope.dependent)
		}
	}
	if !slices.Contains(scope.direct, "overgo/internal/unrelated") {
		t.Fatalf("test source owner lost: %v", scope.direct)
	}
}
