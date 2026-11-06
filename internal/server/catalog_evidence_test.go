package server

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestCatalogEvidenceIndex pins the evidence join: a committed
// benchmark verification surfaces per model with the headline decode
// rate lifted from its evidence document, and the latest committed
// evaluation surfaces per recipe under its derived suite name.
func TestCatalogEvidenceIndex(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	ctx := t.Context()
	store := handler.config.Repository

	weightsPath := filepath.Join(t.TempDir(), "weights.gguf")
	testcheck.NoError(t, os.WriteFile(weightsPath, []byte("catalog evidence weights"), 0o644))
	weights := testcheck.Must(os.Open(weightsPath))(t)
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	weights.Close()
	testcheck.NoError(t, err)
	evidenceData := []byte(`{"summary":{"decode_tokens_per_second_p50":491.35}}`)
	evidence, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(evidenceData))
	testcheck.NoError(t, err)
	record := testcheck.Must(runrecord.NewModelVerification(model, "fixture", []runrecord.CapabilityClaim{{
		Capability: "benchmark", Tier: runrecord.TierCapabilityMeasured,
		Commit: "0123456789abcdef0123456789abcdef01234567",
		WallNS: 220_000_000, PeakDeviceBytes: 512, Evidence: []artifact.ID{evidence},
	}}))(t)
	testcheck.NoError(t, runrecord.CommitVerificationClaim(ctx, store, record, evidenceData, evidence, weightsPath))

	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "catalog-evidence-recipe")
	datasetID := testutil.ArtifactID(t, artifact.KindDataset, "catalog-evidence-dataset")
	runID := testutil.ArtifactID(t, artifact.KindRun, "catalog-evidence-run")
	evaluation := testcheck.Must(runrecord.NewEvaluation(recipeID, runID, datasetID, []runrecord.Metric{
		{Name: "accuracy", Value: 0.42, Direction: runrecord.DirectionMaximize},
	}))(t)
	evaluationContent := testcheck.Must(evaluation.Content())(t)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/catalog-evidence/evaluation", Contents: []artifact.Content{evaluationContent},
	}); err != nil {
		t.Fatal(err)
	}

	index := handler.catalogEvidence(ctx, map[artifact.ID]string{datasetID: "store/mmlu"})
	// The join key is the registered weights location: the claim's model
	// is the weights digest, the catalog's is the manifest, and the
	// on-disk file is the identity both register.
	registered := testcheck.Must(filepath.Abs(weightsPath))(t)
	benchmark, measured := index.BenchmarksByLocation[registered]
	if !measured || benchmark.Tier != string(runrecord.TierCapabilityMeasured) ||
		benchmark.DecodeTokensPerSecond != 491.35 || benchmark.WallNS != 220_000_000 {
		t.Fatalf("benchmark summary = (%+v, %t) keys=%v", benchmark, measured, index.BenchmarksByLocation)
	}
	evals := index.EvaluationsByRecipe[recipeID]
	if len(evals) != 1 || evals[0].Suite != "store/mmlu" || evals[0].Metrics["accuracy"] != 0.42 {
		t.Fatalf("eval summaries = %+v", evals)
	}
}
