package main

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/scratchmodel"
	"overgo/internal/testcheck"
)

func TestPublishArchitectureProfileCatalogCommand(t *testing.T) {
	repository := t.TempDir()
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-repo", repository}, &output))
	store := overgodbtest.OpenReadOnly(t, repository)
	coverage := testcheck.Must(modelrecipe.InspectArchitectureProfileCatalog(t.Context(), store))(t)
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
