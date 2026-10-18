package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelintake"
	"overgo/internal/overgodb"
)

func TestFrozenComponentImportRegistration(t *testing.T) {
	root, _ := registrationFixture(t)
	path := filepath.Join(root, "fixture")
	if err := os.WriteFile(filepath.Join(path, "config.json"), []byte(`{"model_type":"unregistered","architectures":["RemoteModel"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	frozen := filepath.Join(t.TempDir(), "inspection.json")
	if err := inspectModel([]string{"-output", frozen, path}); err != nil {
		t.Fatal(err)
	}
	expected, err := modelintake.InspectImport(t.Context(), path, "")
	if err != nil {
		t.Fatal(err)
	}
	repository := filepath.Join(t.TempDir(), "store")
	args := []string{"-repo", repository, "-inspection", frozen, path}
	if err := registerModels(args); err != nil {
		t.Fatal(err)
	}
	readHead := func() artifact.CommitID {
		t.Helper()
		store, err := overgodb.OpenReadOnly(repository)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		retained, found, err := artifact.ReadContent(t.Context(), store, expected.Content.Descriptor.ID)
		if err != nil || !found || !bytes.Equal(retained.Data, expected.Content.Data) {
			t.Fatalf("retained description differs: %v", err)
		}
		result, err := store.Query(t.Context(), overgodb.Query{Kind: artifact.KindRecipe, MaxResults: 1, Projection: overgodb.ProjectArtifacts})
		if err != nil || result.Matched != 0 {
			t.Fatalf("structural import created recipe: %v", err)
		}
		head, _ := store.Head()
		return head
	}
	head := readHead()
	if err := registerModels(args); err != nil {
		t.Fatal(err)
	}
	if readHead() != head {
		t.Fatal("repeat import changed database")
	}
	if err := os.WriteFile(filepath.Join(path, "config.json"), []byte(`{"model_type":"different","architectures":["RemoteModel"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := registerModels(args); err == nil {
		t.Fatal("changed source was silently re-inspected and registered")
	}
	if readHead() != head {
		t.Fatal("rejected import changed database")
	}
	for _, invalid := range [][]string{
		{"-repo", repository, "-inspection", frozen},
		{"-inspection", frozen, path},
		{"-repo", repository, "-inspection", frozen, "-spec", frozen, path},
		{"-repo", repository, "-inspection", frozen, "-root", root, path},
	} {
		if err := registerModels(invalid); err == nil {
			t.Fatalf("ambiguous or incomplete registration accepted: %v", invalid)
		}
	}
}
