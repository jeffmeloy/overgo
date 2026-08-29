package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMoERoutingBaselineMatrixHasExactPairedAuthorities(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	authorities := moeRoutingPairedAuthorities{
		Model: id(artifact.KindModel, "model"), Dataset: id(artifact.KindDataset, "dataset"),
		Split: id(artifact.KindDatasetShard, "split"), DataOrder: id(artifact.KindEvidence, "data-order"),
		Seed: id(artifact.KindEvidence, "seed"), ComputeBudget: id(artifact.KindEvidence, "compute-budget"),
		Evaluator: id(artifact.KindProfile, "evaluator"), Checkpoint: id(artifact.KindCheckpoint, "checkpoint"),
		Environment: id(artifact.KindEvidence, "environment"), Code: id(artifact.KindEvidence, "code"),
	}
	arms := []moeRoutingBaselineArm{
		{Kind: moeRoutingQuantileBias, Policy: id(artifact.KindRecipe, "quantile-policy"), Recipe: id(artifact.KindRecipe, "quantile-recipe")},
		{Kind: moeRoutingNoBalancing, Policy: id(artifact.KindRecipe, "none-policy"), Recipe: id(artifact.KindRecipe, "none-recipe")},
		{Kind: moeRoutingStaticBias, Policy: id(artifact.KindRecipe, "static-policy"), Recipe: id(artifact.KindRecipe, "static-recipe")},
		{Kind: moeRoutingAuxiliaryLoss, Policy: id(artifact.KindRecipe, "aux-policy"), Recipe: id(artifact.KindRecipe, "aux-recipe")},
	}
	matrix, err := moeRoutingBaselineCodec.New(moeRoutingBaselineMatrix{
		Version: artifact.InitialDocumentVersion, moeRoutingPairedAuthorities: authorities, Arms: arms,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := moeRoutingBaselineCodec.Content(matrix)
	if err != nil || content.Descriptor.ID != matrix.ID || len(moeRoutingBaselineLineage(matrix)) != 18 {
		t.Fatalf("paired matrix = (%+v, %v)", matrix, err)
	}
	missing := matrix
	missing.ID, missing.Arms = artifact.ID{}, missing.Arms[:3]
	if _, err := moeRoutingBaselineCodec.New(missing); err == nil {
		t.Fatal("incomplete baseline matrix admitted")
	}
	duplicate := matrix
	duplicate.ID, duplicate.Arms = artifact.ID{}, append([]moeRoutingBaselineArm(nil), matrix.Arms...)
	duplicate.Arms[0].Policy = duplicate.Arms[1].Policy
	if _, err := moeRoutingBaselineCodec.New(duplicate); err == nil {
		t.Fatal("baseline arms sharing policy authority admitted")
	}
	changedPair := matrix
	changedPair.ID, changedPair.Code = artifact.ID{}, id(artifact.KindEvidence, "different-code")
	changed, err := moeRoutingBaselineCodec.New(changedPair)
	if err != nil || changed.ID == matrix.ID {
		t.Fatal("changed paired authority did not change matrix identity")
	}
}
