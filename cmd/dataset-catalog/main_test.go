package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataset"
	"overgo/internal/overgodb"
)

func TestPublishDatasetCatalogCommand(t *testing.T) {
	repository, legacyRoot, contentRoot := t.TempDir(), t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(legacyRoot, "datasets.log"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(contentRoot, "fixture.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	args := []string{"-repo", repository, "-legacy", legacyRoot, "-root", contentRoot}
	if err := run(args, &output); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	coverage, err := dataset.InspectCatalog(context.Background(), store)
	if err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Registered != 1 || coverage.Available != coverage.Registered ||
		!strings.Contains(output.String(), "changed=true") {
		t.Fatalf("coverage=%+v output=%q", coverage, output.String())
	}
}
