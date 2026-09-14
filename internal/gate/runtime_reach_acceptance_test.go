package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestRuntimeReachBindingsAcceptance(t *testing.T) {
	for name, declarations := range map[string]string{
		"shadowed_owner":     `import realos "os";type paths struct{};func(paths)Executable()string{return os.Getenv("INPUT")};func readInput()([]byte,error){os:=paths{};p:=os.Executable();return realos.ReadFile(p)}`,
		"escaped_path_alias": `func replace(p *string){*p=os.Getenv("INPUT")};func readInput()([]byte,error){p:=os.TempDir();replace(&p);q:=p;r:=q;return os.ReadFile(r)}`,
		"branch_alias":       `func readInput()([]byte,error){p:=os.TempDir();q:=p;if os.Getenv("INPUT")!=""{p=os.Getenv("INPUT")};q=p;return os.ReadFile(q)}`,
		"caller_annotation":  "//overgo:runtime-inputs caller\nfunc load(p string)([]byte,error){return os.ReadFile(p)};func readInput()([]byte,error){return load(os.Getenv(\"INPUT\"))}",
		"temporary_method":   `type paths struct{};func(paths)TempDir()string{return os.Getenv("INPUT")};func readInput()([]byte,error){return os.ReadFile(paths{}.TempDir())}`,
		"escaped_path":       `func replace(p *string){*p=os.Getenv("INPUT")};func readInput()([]byte,error){p:=os.TempDir();replace(&p);return os.ReadFile(p)}`,
		"reassigned":         `func readInput()([]byte,error){ read:=func(string)([]byte,error){return nil,nil}; read=os.ReadFile; return read(os.Getenv("INPUT")) }`,
		"conditional":        `func readInput()([]byte,error){ read:=func(string)([]byte,error){return nil,nil}; if os.Getenv("INPUT")!="" {read=os.ReadFile}; return read(os.Getenv("INPUT")) }`,
		"returned":           `func reader()func(string)([]byte,error){return os.ReadFile}; func readInput()([]byte,error){return reader()(os.Getenv("INPUT"))}`,
		"callback":           `func invoke(read func(string)([]byte,error))([]byte,error){return read(os.Getenv("INPUT"))}; func readInput()([]byte,error){return invoke(os.ReadFile)}`,
		"interface":          `type loader interface{Fetch(string)([]byte,error)}; type disk struct{read func(string)([]byte,error)}; func(d disk)Fetch(p string)([]byte,error){return d.read(p)}; func readInput()([]byte,error){var d loader=disk{os.ReadFile};return d.Fetch(os.Getenv("INPUT"))}`,
	} {
		t.Run(name, func(t *testing.T) {
			g := runtimeReaderFixture(t)
			source := "package reader\nimport(\"os\";\"strings\")\n" + declarations + "\nfunc Value()int{b,e:=readInput();if e==nil&&strings.TrimSpace(string(b))==\"1\"{return 1};return -1}\n"
			if err := os.WriteFile(filepath.Join(g.repo, "internal", "reader", "reader.go"), []byte(source), 0o644); err != nil {
				t.Fatal(err)
			}
			input := filepath.Join(g.repo, "docs", "config.txt")
			t.Setenv("INPUT", input)
			graph, err := g.inputGraph()
			if err != nil {
				t.Fatal(err)
			}
			packages := []string{"overgo/internal/readerclient", "overgo/internal/isolated"}
			before, err := packageInputIdentities(graph, packages)
			if err != nil {
				t.Fatal(err)
			}
			if out, err := command(g.repo, "go", "test", "./internal/readerclient", "./internal/isolated", "-count=1"); err != nil {
				t.Fatalf("baseline: %v\n%s", err, out)
			}
			if err := os.WriteFile(input, []byte("2\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			if out, err := command(g.repo, "go", "test", "./internal/readerclient", "-count=1"); err == nil {
				t.Fatalf("seeded consumer did not fail: %s", out)
			}
			after, err := packageInputIdentities(graph, packages)
			if err != nil {
				t.Fatal(err)
			}
			g.paths = []string{"docs/config.txt"}
			scope, err := g.deriveTestScope()
			if err != nil {
				t.Fatal(err)
			}
			selected := scope.selected()
			if !slices.Contains(selected, packages[0]) || before[packages[0]] == after[packages[0]] {
				t.Errorf("failing indirect consumer escaped: selected=%v invalidated=%t", selected, before[packages[0]] != after[packages[0]])
			}
			if slices.Contains(selected, packages[1]) || before[packages[1]] != after[packages[1]] {
				t.Errorf("isolated consumer lost reusable evidence: selected=%v", selected)
			}
		})
	}
}

func TestRuntimeBoundTemporarySources(t *testing.T) {
	for name, source := range map[string]string{
		"testing":         "package probe\nimport(\"os\";\"path/filepath\";\"testing\")\nfunc fixture(t *testing.T){p:=filepath.Join(t.TempDir(),\"fixture\");_,_=os.ReadFile(p)}",
		"testing_alias":   "package probe\nimport(myos \"os\";test \"testing\")\nfunc fixture(t *test.T){_,_=myos.ReadFile(t.TempDir())}",
		"flags_and_types": "package probe\nimport \"os\"\nvar _ *os.File\nfunc fixture(){_,_=os.OpenFile(\"fixture\",os.O_RDONLY,0600)}",
	} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			if err := os.WriteFile(filepath.Join(root, "probe.go"), []byte(source), 0o644); err != nil {
				t.Fatal(err)
			}
			inputs, err := classifyRuntimeInputs(root, root, []string{"probe.go"})
			if err != nil || !inputs.confined() {
				t.Fatalf("bound inputs became opaque: %+v %v", inputs, err)
			}
		})
	}
}
