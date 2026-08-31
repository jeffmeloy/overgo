package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestTrainingEvidenceQueryPlanLoadsSummariesOnly(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	targetModel := id(artifact.KindModel, "query target model")
	parents := []artifact.ID{
		id(artifact.KindRun, "query run a"), id(artifact.KindRun, "query run b"),
		targetModel, id(artifact.KindModel, "query other model"),
		id(artifact.KindDataset, "query dataset"), id(artifact.KindDatasetShard, "query split"),
		id(artifact.KindRecipe, "query recipe"), id(artifact.KindEvidence, "query code"),
		id(artifact.KindCheckpoint, "query checkpoint"), id(artifact.KindRecipe, "query policy"),
	}
	descriptors := make([]artifact.Descriptor, len(parents))
	for index, parent := range parents {
		descriptors[index] = artifact.Descriptor{ID: parent}
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/router-query/parents", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	for index, model := range []artifact.ID{targetModel, parents[3]} {
		observation, err := NewMoERouterObservation(MoERouterObservation{
			Run: parents[index], Model: model, Dataset: parents[4], Split: parents[5], Recipe: parents[6],
			Code: parents[7], Checkpoint: parents[8], Policy: parents[9], Step: uint64(index),
			Rows: 1, Experts: 2, TopK: 1, Selections: []uint32{1}, CombineWeights: []float32{1},
			Accepted: []bool{true}, Margins: []MoERouterMargin{{Observed: true}},
		})
		if err != nil {
			t.Fatal(err)
		}
		chunk, err := NewMoERouterObservationChunk([]MoERouterObservation{observation}, artifact.ID{})
		if err != nil {
			t.Fatal(err)
		}
		coverage, err := NewMoERouterObservationCoverage(chunk, observation.Step, 1, []uint32{0})
		if err != nil {
			t.Fatal(err)
		}
		batch, err := coverage.Batch(t.Context(), store, chunk)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	}
	children, err := store.Children(t.Context(), targetModel)
	if err != nil || len(children) != 1 {
		t.Fatalf("target model children = %+v err=%v", children, err)
	}

	summaries, err := QueryMoERouterObservationCoverage(t.Context(), store, MoERouterObservationQuery{
		Model: targetModel, Dataset: parents[4], Limit: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if summaries.Plan.Index != "seed" || summaries.Plan.Inspected != 2 || summaries.Plan.Matched != 1 ||
		summaries.Plan.Returned != 1 || summaries.Plan.Loaded != 1 ||
		len(summaries.Summaries) != 1 || len(summaries.Chunks) != 0 ||
		summaries.Summaries[0].Model != targetModel || summaries.Work.SummariesInspected != 1 ||
		summaries.Work.SummariesMatched != 1 || summaries.Work.SummariesReturned != 1 ||
		summaries.Work.SummaryBlobsLoaded != 1 || summaries.Work.SummaryBytesLoaded == 0 ||
		summaries.Work.RawChunksLoaded != 0 || summaries.Work.RawBytesLoaded != 0 {
		t.Fatalf("summary-only router query = %+v", summaries)
	}

	raw, err := QueryMoERouterObservationCoverage(t.Context(), store, MoERouterObservationQuery{
		Model: targetModel, IncludeSamples: true, Limit: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(raw.Chunks) != 1 || len(raw.Chunks[0].Observations) != 1 ||
		raw.Work.RawChunksLoaded != 1 || raw.Work.RawBytesLoaded != raw.Summaries[0].ObservationBytes {
		t.Fatalf("sample router query = %+v", raw)
	}
	head, err := QueryMoERouterObservationCoverage(t.Context(), store, MoERouterObservationQuery{
		Head: true, Model: parents[3], Limit: 1,
	})
	if err != nil || head.Plan.Index != "seed" || len(head.Summaries) != 1 || head.Summaries[0].Model != parents[3] ||
		head.Work.RawChunksLoaded != 0 {
		t.Fatalf("head router query = %+v err=%v", head, err)
	}
}
