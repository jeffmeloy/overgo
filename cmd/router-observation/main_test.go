package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestRunPublishesRouterObservation(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	fixture := runrecord.MoERouterObservation{
		Run: id(artifact.KindRun, "run"), Model: id(artifact.KindModel, "model"), Dataset: id(artifact.KindDataset, "dataset"),
		Split: id(artifact.KindDatasetShard, "split"), Recipe: id(artifact.KindRecipe, "recipe"), Code: id(artifact.KindEvidence, "code"),
		Checkpoint: id(artifact.KindCheckpoint, "checkpoint"), Policy: id(artifact.KindRecipe, "policy"),
		Rows: 1, Experts: 2, TopK: 1, Selections: []uint32{1}, CombineWeights: []float32{1}, Accepted: []bool{true},
		Margins: []runrecord.MoERouterMargin{{Observed: true}},
	}
	data, err := json.Marshal(fixture)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	input := filepath.Join(root, "observation.json")
	if err := os.WriteFile(input, data, 0o600); err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(root, "store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	parents := []artifact.Descriptor{
		{ID: fixture.Run}, {ID: fixture.Model}, {ID: fixture.Dataset}, {ID: fixture.Split},
		{ID: fixture.Recipe}, {ID: fixture.Code}, {ID: fixture.Checkpoint}, {ID: fixture.Policy},
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/router-observation/parents", Artifacts: parents}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := run([]string{"-input", input, "-store", storePath}, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "rows=1 selections=1") {
		t.Fatalf("output = %q", output.String())
	}
	store, err = overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()
	coverageID, found, err := store.ResolveAlias(t.Context(), runrecord.MoERouterObservationCoverageAlias)
	if err != nil || !found {
		t.Fatalf("resolve coverage alias: found=%v err=%v", found, err)
	}
	coverage, chunk, err := runrecord.RequireMoERouterObservationCoverage(t.Context(), store, coverageID)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := runrecord.NewMoERouterObservation(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if coverage.ObservationCount != 1 || len(chunk.Observations) != 1 || chunk.Observations[0].ID != expected.ID {
		t.Fatalf("coverage = %#v chunk = %#v", coverage, chunk)
	}
}
