package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/libraryintake"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
)

func TestComponentImportInspectionRoutes(t *testing.T) {
	root, _ := registrationFixture(t)
	path := filepath.Join(root, "fixture")
	if err := os.WriteFile(filepath.Join(path, "config.json"), []byte(`{"model_type":"unregistered","architectures":["RemoteModel"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(t.TempDir(), "inspection.json")
	if err := inspectModel([]string{"-output", output, path}); err != nil {
		t.Fatal(err)
	}
	expected, err := modelintake.InspectImport(t.Context(), path, "")
	if err != nil {
		t.Fatal(err)
	}
	actual, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(bytes.TrimSpace(actual), expected.Content.Data) {
		t.Fatal("CLI does not use canonical inspection")
	}
	model, projector, err := libraryintake.ModelFiles(path, "")
	if err != nil || model != path || projector != "" {
		t.Fatalf("library directory route: %s %s %v", model, projector, err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for range 2 {
		receipt, err := libraryintake.Register(t.Context(), store, model, projector)
		if err != nil {
			t.Fatal(err)
		}
		if receipt["inspection"] != expected.Content.Descriptor.ID || receipt["recipe"] != nil || receipt["activation"] != nil {
			t.Fatalf("source import acquired serving authority: %+v", receipt)
		}
	}
	retained, found, err := artifact.ReadContent(t.Context(), store, expected.Content.Descriptor.ID)
	if err != nil || !found || !bytes.Equal(retained.Data, expected.Content.Data) {
		t.Fatalf("stored inspection differs: %v", err)
	}
	if _, _, err := libraryintake.ModelFiles(path, "projector.gguf"); err == nil {
		t.Fatal("unbound projector accepted for HF directory")
	}
}
