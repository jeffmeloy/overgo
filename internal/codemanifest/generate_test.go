package codemanifest

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func TestGenerateSymbolIdentityAndReferenceGraph(t *testing.T) {
	root := t.TempDir()
	writeGeneratorFixture(t, root, "internal/example/apply.go", "package example\nfunc Apply(value int) int { return value + 1 }\n")
	writeGeneratorFixture(t, root, "internal/example/run.go", "package example\nfunc Run() int { return Apply(2) }\n")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{
		Context: "linux/amd64", Root: root,
		Files:    map[string]bool{"internal/example/apply.go": true, "internal/example/run.go": true},
		Packages: map[string]string{"internal/example/apply.go": "overgo/internal/example", "internal/example/run.go": "overgo/internal/example"},
	}
	manifest, err := Generate(snapshot, []gosource.BuildSelection{selection}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(manifest.Symbols) != 2 || len(manifest.References) != 1 {
		t.Fatalf("manifest symbols=%d references=%d", len(manifest.Symbols), len(manifest.References))
	}
	ref := manifest.References[0]
	if ref.From.Name != "Run" || ref.To.Name != "Apply" || ref.Kind != ReferenceCall {
		t.Fatalf("reference = %+v", ref)
	}
	if manifest.Symbols[0].SignatureSHA256 == "" || manifest.Symbols[0].BodySHA256 == "" {
		t.Fatal("function fingerprints are absent")
	}
	if manifest.ID.Kind().String() != "profile" {
		t.Fatalf("manifest identity = %s", manifest.ID)
	}
}

func TestGenerateSeparatesSignatureAndBody(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/run.go"
	writeGeneratorFixture(t, root, name, "package example\nfunc Run() int { return 1 }\n")
	first := generateSingleFile(t, root, name)
	writeGeneratorFixture(t, root, name, "package example\nfunc Run() int { return 2 }\n")
	second := generateSingleFile(t, root, name)
	if first.Symbols[0].SignatureSHA256 != second.Symbols[0].SignatureSHA256 || first.Symbols[0].BodySHA256 == second.Symbols[0].BodySHA256 {
		t.Fatalf("signature/body separation failed: first=%+v second=%+v", first.Symbols[0], second.Symbols[0])
	}
}

func generateSingleFile(t *testing.T, root, name string) Manifest {
	t.Helper()
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := Generate(snapshot, []gosource.BuildSelection{{
		Context: "linux/amd64", Root: root, Files: map[string]bool{name: true}, Packages: map[string]string{name: "overgo/internal/example"},
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return manifest
}

func writeGeneratorFixture(t *testing.T, root, name, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
