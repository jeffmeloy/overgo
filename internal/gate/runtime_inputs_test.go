package gate

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestRuntimeInputsUnsafeClassifications pins three inputs the classifier
// once treated as having no external reach: a path taken from an operator
// argument other than the binary's own, a local first assigned a literal and
// then an environment value, and reads and executions through an aliased
// owner import. Each keeps the broad binding or names its repository input;
// the binary's own argument and a dot import of an owner stay classified.
func TestRuntimeInputsUnsafeClassifications(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	write := func(pkg, name, source string) string {
		dir := filepath.Join(root, "internal", pkg)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	classify := func(dir string, files ...string) runtimeInputs {
		inputs, err := classifyRuntimeInputs(root, dir, files)
		if err != nil {
			t.Fatal(err)
		}
		return inputs
	}
	unnamed := func(reasons []string) bool {
		return slices.ContainsFunc(reasons, func(reason string) bool { return strings.Contains(reason, "does not name") })
	}

	operator := write("operator", "main.go", "package operator\n\nimport \"os\"\n\nfunc Read() ([]byte, error) { return os.ReadFile(os.Args[1]) }\n")
	if inputs := classify(operator, "main.go"); !unnamed(inputs.dynamic) || inputs.confined() {
		t.Fatalf("operator argument = %+v, want an unnamed input", inputs)
	}
	self := write("self", "main.go", "package self\n\nimport \"os\"\n\nfunc Read() ([]byte, error) { return os.ReadFile(os.Args[0]) }\n")
	if inputs := classify(self, "main.go"); !inputs.confined() {
		t.Fatalf("the binary's own path = %+v, want confined", inputs)
	}

	reassigned := write("reassigned", "main.go", "package reassigned\n\nimport \"os\"\n\nfunc Read() ([]byte, error) {\n\tpath := \"fixture.txt\"\n\tpath = os.Getenv(\"OVERGO_PROBE_PATH\")\n\treturn os.ReadFile(path)\n}\n")
	if inputs := classify(reassigned, "main.go"); !unnamed(inputs.dynamic) || inputs.confined() {
		t.Fatalf("local reassigned from the environment = %+v, want an unnamed input", inputs)
	}
	derived := write("derived", "main.go", "package derived\n\nimport \"os\"\n\nfunc Read(name string) ([]byte, error) {\n\tpath := name\n\tpath = os.Getenv(\"OVERGO_PROBE_PATH\")\n\treturn os.ReadFile(path)\n}\n")
	if inputs := classify(derived, "main.go"); !unnamed(inputs.dynamic) {
		t.Fatalf("parameter reassigned from the environment = %+v, want an unnamed input, not the caller's reach", inputs)
	}

	aliased := write("aliased", "main.go", "package aliased\n\nimport (\n\tmyexec \"os/exec\"\n\tmyos \"os\"\n)\n\nfunc Run() error {\n\tif _, err := myos.ReadFile(\"../../docs/protocol.txt\"); err != nil {\n\t\treturn err\n\t}\n\treturn myexec.Command(\"../../cmd/tool\").Run()\n}\n")
	inputs := classify(aliased, "main.go")
	if !slices.Equal(inputs.files, []string{"docs/protocol.txt"}) || !slices.Equal(inputs.commands, []string{"cmd/tool"}) || len(inputs.dynamic) != 0 {
		t.Fatalf("aliased owner imports = %+v, want the named document and command", inputs)
	}
	aliasedEnvironment := write("aliasedenv", "main.go", "package aliasedenv\n\nimport myos \"os\"\n\nfunc Read() ([]byte, error) { return myos.ReadFile(myos.Getenv(\"OVERGO_PROBE_PATH\")) }\n")
	if inputs := classify(aliasedEnvironment, "main.go"); !unnamed(inputs.dynamic) {
		t.Fatalf("aliased read of an environment path = %+v, want an unnamed input", inputs)
	}
	dotted := write("dotted", "main.go", "package dotted\n\nimport . \"os\"\n\nfunc Read() ([]byte, error) { return ReadFile(\"fixture.txt\") }\n")
	if inputs := classify(dotted, "main.go"); inputs.confined() || !slices.ContainsFunc(inputs.dynamic, func(reason string) bool { return strings.Contains(reason, "without a name") }) {
		t.Fatalf("dot-imported owner = %+v, want the broad binding", inputs)
	}

	// Owner review 2026-09-10: a mixed assignment holding an environment
	// path beside a literal made both names safe; each target takes the
	// safety of its own value.
	mixed := write("mixed", "main.go", "package mixed\n\nimport \"os\"\n\nfunc Read() ([]byte, error) {\n\tpath, mode := os.Getenv(\"OVERGO_PROBE_PATH\"), \"fixture\"\n\t_ = mode\n\treturn os.ReadFile(path)\n}\n")
	if inputs := classify(mixed, "main.go"); !unnamed(inputs.dynamic) || inputs.confined() {
		t.Fatalf("mixed assignment with an environment path = %+v, want an unnamed input", inputs)
	}
	pairedLiteral := write("pairedliteral", "main.go", "package pairedliteral\n\nimport \"os\"\n\nfunc Read() ([]byte, error) {\n\tmode, path := os.Getenv(\"OVERGO_PROBE_MODE\"), \"fixture.txt\"\n\t_ = mode\n\treturn os.ReadFile(path)\n}\n")
	if inputs := classify(pairedLiteral, "main.go"); len(inputs.dynamic) != 0 {
		t.Fatalf("literal target beside an environment value = %+v, want its own literal safety", inputs)
	}
	multi := write("multi", "main.go", "package multi\n\nimport \"os\"\n\nfunc pair() (string, string) { return os.Getenv(\"OVERGO_PROBE_PATH\"), \"fixture\" }\n\nfunc Read() ([]byte, error) {\n\tpath, mode := pair()\n\t_ = mode\n\treturn os.ReadFile(path)\n}\n")
	if inputs := classify(multi, "main.go"); !unnamed(inputs.dynamic) {
		t.Fatalf("multi-value assignment = %+v, want an unnamed input", inputs)
	}
	// Owner review 2026-09-10: go run with an environment-derived target was
	// confined because only the tool's name was read; the target is the
	// program reached.
	goRunEnvironment := write("gorunenv", "main.go", "package gorunenv\n\nimport (\n\t\"os\"\n\t\"os/exec\"\n)\n\nfunc Run() error { return exec.Command(\"go\", \"run\", os.Getenv(\"OVERGO_PROBE_TARGET\")).Run() }\n")
	if inputs := classify(goRunEnvironment, "main.go"); inputs.confined() || !slices.ContainsFunc(inputs.dynamic, func(reason string) bool {
		return strings.Contains(reason, "executes a program the source does not name")
	}) {
		t.Fatalf("go run of an environment target = %+v, want an unnamed program", inputs)
	}
	goRunLiteral := write("gorunliteral", "main.go", "package gorunliteral\n\nimport (\n\t\"context\"\n\t\"os/exec\"\n)\n\nfunc Run(ctx context.Context) error { return exec.CommandContext(ctx, \"go\", \"run\", \"../../cmd/tool\", \"-flag\").Run() }\n")
	if inputs := classify(goRunLiteral, "main.go"); !slices.Equal(inputs.commands, []string{"cmd/tool"}) || len(inputs.dynamic) != 0 {
		t.Fatalf("go run of a repository command = %+v, want the named command", inputs)
	}
}
