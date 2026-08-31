package trainingworkflow

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func TestTrainingEvidenceDeltaIsSummaryOnly(t *testing.T) {
	stubExecutableCodeCommit(t, strings.Repeat("bc", 20))
	root, model, dataset, store, recipeID := moeWorkflowFixture(t)
	anchorHead, anchorSequence := store.Head()
	result, err := Execute(t.Context(), Request{
		Repository: store, Observations: store, Recipe: recipeID,
		ModelDirectory: model, DatasetPath: dataset,
		OutputDirectory: filepath.Join(root, "delta-evidence"), Steps: 2, Host: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	coverage, chunk, err := runrecord.RequireMoERouterObservationCoverage(
		t.Context(), store, result.RouterObservationCoverage,
	)
	if err != nil {
		t.Fatal(err)
	}
	delta, resync, err := store.DeltasSince(
		t.Context(), anchorHead, anchorSequence, overgodb.ProjectionContractVersion(), 8,
	)
	if err != nil || resync {
		t.Fatalf("training delta = %+v resync=%t err=%v", delta, resync, err)
	}
	if !slices.Contains(delta.Contents, coverage.ID) || !slices.Contains(delta.Contents, chunk.ID) {
		t.Fatalf("training delta lost summary or lazy payload identity: %+v", delta.Contents)
	}
	for _, observation := range chunk.Observations {
		if slices.Contains(delta.Contents, observation.ID) {
			t.Fatalf("training delta copied child observation identity %s", observation.ID)
		}
	}
	raw, err := chunk.Content()
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(delta)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(wire, raw.Data) || bytes.Contains(wire, []byte(`"observations"`)) ||
		bytes.Contains(wire, []byte(`"selections"`)) {
		t.Fatalf("training delta embedded raw router payload: %s", wire)
	}
}
