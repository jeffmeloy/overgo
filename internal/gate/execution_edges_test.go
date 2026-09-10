package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestDeclaredExecutionEdgesBoundSelection pins the runtime-input analysis
// and the selection it drives: a package whose test runs a literal repository
// command binds that command's package, a package that reads only temporary
// files and runs external tools is confined and leaves the scope of an
// unrelated command change, a package that runs a program its source does
// not name keeps the broad binding and is reported, and a repository path
// literal binds the file it names.
func TestDeclaredExecutionEdgesBoundSelection(t *testing.T) {
	g := scopeCompilerFixture(t)
	write := func(name, content string) {
		path := filepath.Join(g.repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("cmd/tool/main.go", "package main\nimport \"fmt\"\nfunc main() { fmt.Print(1) }\n")
	write("internal/runner/runner_test.go", "package runner\nimport (\"os/exec\"; \"testing\")\nfunc TestRun(t *testing.T) { out, err := exec.CommandContext(t.Context(), \"go\", \"run\", \"../../cmd/tool\").CombinedOutput(); if err != nil || string(out) != \"1\" { t.Fatal(string(out), err) } }\n")
	write("internal/confined/confined.go", "package confined\nconst Value = 1\n")
	write("internal/confined/confined_test.go", "package confined\nimport (\"os\"; \"os/exec\"; \"path/filepath\"; \"testing\")\nfunc TestRead(t *testing.T) { name := filepath.Join(t.TempDir(), \"note.txt\"); if err := os.WriteFile(name, []byte(\"x\"), 0o644); err != nil { t.Fatal(err) }; if _, err := os.ReadFile(name); err != nil { t.Fatal(err) }; _ = exec.Command(os.Args[0], \"-test.run=none\"); _ = exec.Command(\"git\", \"--version\") }\n")
	write("internal/dynamic/dynamic_test.go", "package dynamic\nimport (\"os\"; \"os/exec\"; \"testing\")\nfunc TestRun(t *testing.T) { program := os.Getenv(\"TOOL\"); if program == \"\" { t.Skip() }; _ = exec.Command(program) }\n")
	write("internal/reader/reader_test.go", "package reader\nimport (\"os\"; \"testing\")\nfunc TestRead(t *testing.T) { if _, err := os.ReadFile(\"../../docs/protocol.txt\"); err != nil { t.Skip(err) } }\n")
	write("docs/protocol.txt", "1")
	runGitFixture(t, g.repo, "add", ".")
	g.packageGraph = nil
	graph, err := g.inputGraph()
	if err != nil {
		t.Fatal(err)
	}
	classify := func(directory string) runtimeInputs {
		t.Helper()
		for _, node := range graph.nodes {
			if node.ForTest != "" || filepath.ToSlash(node.Dir) != filepath.ToSlash(filepath.Join(g.repo, filepath.FromSlash(directory))) {
				continue
			}
			inputs, err := classifyRuntimeInputs(g.repo, node.Dir, packageSources(node))
			if err != nil {
				t.Fatal(err)
			}
			return inputs
		}
		t.Fatalf("package %s not in graph", directory)
		return runtimeInputs{}
	}
	if inputs := classify("internal/runner"); !slices.Equal(inputs.commands, []string{"cmd/tool"}) || len(inputs.dynamic) != 0 {
		t.Fatalf("runner inputs = %+v", inputs)
	}
	if inputs := classify("internal/confined"); !inputs.confined() {
		t.Fatalf("confined inputs = %+v", inputs)
	}
	if inputs := classify("internal/dynamic"); len(inputs.testDynamic) != 1 || !strings.Contains(inputs.testDynamic[0], "does not name") || len(inputs.dynamic) != 0 {
		t.Fatalf("dynamic inputs = %+v", inputs)
	}
	if inputs := classify("internal/reader"); !slices.Equal(inputs.testFiles, []string{"docs/protocol.txt"}) || len(inputs.files) != 0 {
		t.Fatalf("reader inputs = %+v", inputs)
	}

	g.paths = []string{"cmd/tool/main.go"}
	scope, err := g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	selected := append(slices.Clone(scope.direct), scope.dependent...)
	for _, want := range []string{"overgo/internal/runner", "overgo/internal/dynamic"} {
		if !slices.Contains(selected, want) {
			t.Fatalf("command change did not select %s: %v", want, selected)
		}
	}
	for _, unwanted := range []string{"overgo/internal/confined", "overgo/internal/reader", "overgo/internal/client"} {
		if slices.Contains(selected, unwanted) {
			t.Fatalf("command change selected the confined package %s: %v", unwanted, selected)
		}
	}
	if !slices.Contains(scope.opaqueRuntimeInputs, "overgo/internal/dynamic") || slices.Contains(scope.opaqueRuntimeInputs, "overgo/internal/confined") {
		t.Fatalf("opaque runtime inputs = %v", scope.opaqueRuntimeInputs)
	}

	g.paths = []string{"docs/protocol.txt"}
	scope, err = g.deriveTestScope()
	if err != nil {
		t.Fatal(err)
	}
	selected = append(slices.Clone(scope.direct), scope.dependent...)
	if !slices.Contains(selected, "overgo/internal/reader") || slices.Contains(selected, "overgo/internal/confined") {
		t.Fatalf("file change selection = %v", selected)
	}
}
