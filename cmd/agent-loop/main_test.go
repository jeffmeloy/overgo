package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

const cliSessionSteps = 1

func TestAgentLoopProposesGatedStep(t *testing.T) {
	repository := t.TempDir()
	store, err := overgodb.Open(repository)
	if err != nil {
		t.Fatal(err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "cli-loop-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "cli-loop-model")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "cli-loop/identity", Artifacts: []artifact.Descriptor{{ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"seen":true}`))
	}))
	defer server.Close()
	inspect, err := agenttool.NewManual(agenttool.Manual{
		Name: "probe.read", Description: "Read state for the CLI fixture.",
		Effect:    agenttool.EffectInspection,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(context.Background(), store, []agenttool.Manual{inspect}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := run([]string{
		"-repo", repository, "-session", "cli-1", "-tool", "probe.read",
		"-recipe", recipeID.String(), "-model", modelID.String(),
		"-max-steps", strconv.Itoa(cliSessionSteps),
	}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"seen":true`) {
		t.Fatalf("loop output = %q", out.String())
	}
}
