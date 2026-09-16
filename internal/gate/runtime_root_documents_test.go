package gate

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestRootDocumentLiteralNamesTheDocument pins the classification of a bare
// repository-root file name: a reader that joins the root with the literal
// name of a file at the root names that document, so the document binds to
// its readers; the same literal for a name that is no root file stays a
// pending tree name, and a literal in a message names nothing.
func TestRootDocumentLiteralNamesTheDocument(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "compatibility.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	write := func(pkg, source string) string {
		dir := filepath.Join(root, "cmd", pkg)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(source), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	reader := write("reader", "package main\n\nimport (\n\t\"os\"\n\t\"path/filepath\"\n)\n\nconst manifestPath = \"compatibility.json\"\n\nfunc load(root string) ([]byte, error) { return os.ReadFile(filepath.Join(root, manifestPath)) }\n\nfunc main() {}\n")
	inputs, err := classifyRuntimeInputs(root, reader, []string{"main.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(inputs.files, "compatibility.json") {
		t.Fatalf("root document read by literal is not named: %+v", inputs)
	}
	message := write("message", "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"compatibility.json\") }\n")
	inputs, err = classifyRuntimeInputs(root, message, []string{"main.go"})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(inputs.files, "compatibility.json") {
		t.Fatalf("a message literal named the root document: %+v", inputs)
	}
	absent := write("absent", "package main\n\nimport (\n\t\"os\"\n\t\"path/filepath\"\n)\n\nfunc load(root string) ([]byte, error) { return os.ReadFile(filepath.Join(root, \"absent.json\")) }\n\nfunc main() {}\n")
	inputs, err = classifyRuntimeInputs(root, absent, []string{"main.go"})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(inputs.files, "absent.json") {
		t.Fatalf("a name that is no root file was named as a document: %+v", inputs)
	}
	// Root Markdown is documentation whoever reads it: a README reader does
	// not make a README change select its suite.
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("# fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	readme := write("readme", "package main\n\nimport (\n\t\"os\"\n\t\"path/filepath\"\n)\n\nfunc load(root string) ([]byte, error) { return os.ReadFile(filepath.Join(root, \"README.md\")) }\n\nfunc main() {}\n")
	inputs, err = classifyRuntimeInputs(root, readme, []string{"main.go"})
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(inputs.files, "README.md") {
		t.Fatalf("root Markdown was named as a document: %+v", inputs)
	}
	// A generator hands its document's name to the shared output owner.
	if err := os.WriteFile(filepath.Join(root, "SBOM.cdx.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	generator := write("generator", "package main\n\nimport (\n\t\"os\"\n\n\t\"overgo/internal/clioptions\"\n)\n\nconst sbomPath = \"SBOM.cdx.json\"\n\nfunc main() {\n\tdata := generate()\n\t_ = clioptions.OutputGenerated(data, sbomPath, false, true, \"stale\", os.Stdout)\n}\n\nfunc generate() []byte { return nil }\n")
	inputs, err = classifyRuntimeInputs(root, generator, []string{"main.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(inputs.files, "SBOM.cdx.json") {
		t.Fatalf("a generator's output document is not named: %+v", inputs)
	}
}
