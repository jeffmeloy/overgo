package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

func TestPublishArchitectureProfileCatalogCommand(t *testing.T) {
	repository := t.TempDir()
	var output bytes.Buffer
	if err := run([]string{"-repo", repository}, &output); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	coverage, err := modelrecipe.InspectArchitectureProfileCatalog(context.Background(), store)
	if err != nil {
		t.Fatal(err)
	}
	if !coverage.Complete || coverage.Registered == 0 || coverage.Published != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
	if !strings.Contains(output.String(), "changed=true") {
		t.Fatalf("output = %q", output.String())
	}
}
