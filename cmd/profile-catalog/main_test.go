package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/scratchmodel"
	"overgo/internal/testcheck"
)

func TestPublishArchitectureProfileCatalogCommand(t *testing.T) {
	repository := t.TempDir()
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository}, &output))
	store, err := overgodb.OpenReadOnly(repository)
	testcheck.NoError(t, err)
	defer store.Close()
	coverage, err := modelrecipe.InspectArchitectureProfileCatalog(t.Context(), store)
	testcheck.NoError(t, err)
	if !coverage.Complete || coverage.Registered == 0 || coverage.Published != coverage.Registered {
		t.Fatalf("coverage = %+v", coverage)
	}
	if _, err := scratchmodel.ResolveActiveDerivationProfile(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "changed=true") {
		t.Fatalf("output = %q", output.String())
	}
}
