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
	"overgo/internal/testcheck"
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
	data := testcheck.Must(json.Marshal(fixture))(t)
	root := t.TempDir()
	input := filepath.Join(root, "observation.json")
	testcheck.NoError(t, os.WriteFile(input, data, 0o600))
	storePath := filepath.Join(root, "store")
	store := testcheck.Must(overgodb.Create(storePath))(t)
	parents := []artifact.Descriptor{
		{ID: fixture.Run}, {ID: fixture.Model}, {ID: fixture.Dataset}, {ID: fixture.Split},
		{ID: fixture.Recipe}, {ID: fixture.Code}, {ID: fixture.Checkpoint}, {ID: fixture.Policy},
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/router-observation/parents", Artifacts: parents}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	var output bytes.Buffer
	testcheck.NoError(t, run([]string{"-input", input, "-store", storePath}, &output))
	if !strings.Contains(output.String(), "rows=1 selections=1") {
		t.Fatalf("output = %q", output.String())
	}
	store = testcheck.Must(overgodb.Create(storePath))(t)
	defer func() { _ = store.Close() }()
	coverageID, found, err := store.ResolveAlias(t.Context(), runrecord.MoERouterObservationCoverageAlias)
	if err != nil || !found {
		t.Fatalf("resolve coverage alias: found=%v err=%v", found, err)
	}
	coverage, chunk, err := runrecord.RequireMoERouterObservationCoverage(t.Context(), store, coverageID)
	testcheck.NoError(t, err)
	expected := testcheck.Must(runrecord.NewMoERouterObservation(fixture))(t)
	if coverage.ObservationCount != 1 || len(chunk.Observations) != 1 || chunk.Observations[0].ID != expected.ID {
		t.Fatalf("coverage = %#v chunk = %#v", coverage, chunk)
	}
}
