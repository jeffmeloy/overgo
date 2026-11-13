package main

import (
	"io"
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestASmokeWithNoBudgetRunsPastDiscovery holds validate -smoke to its default of no limit (SQA-398): over a store
// that serves one model, a smoke given no -budget goes past discovery, as the DNA re-acquisition's command, which
// passes none, must. It had refused there for want of a positive -budget, so every smoke over a store that serves a
// model failed before it checked anything; this one fails later, at the oracle declarations the fixture leaves out.
func TestASmokeWithNoBudgetRunsPastDiscovery(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	store := testcheck.Must(overgodb.Create(repository))(t)
	model := testutil.HermeticLlamaGGUF(t, 64)
	testcheck.NoError(t, servingtest.PublishActiveGGUFWithPolicy(t.Context(), store, model, recipe.PlacementHost, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference))
	testcheck.NoError(t, store.Close())
	err := runArgs(t.Context(), []string{"-smoke", "-repo", repository}, io.Discard)
	if err == nil || strings.Contains(err.Error(), "-budget") || strings.Contains(err.Error(), "outcome=empty") {
		t.Fatalf("a smoke given no -budget over a store that serves a model ended %v, want it past discovery", err)
	}
}
