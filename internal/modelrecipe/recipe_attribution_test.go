package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func attributionSide(t *testing.T, recipeName string) RecipeEvaluationSide {
	t.Helper()
	return RecipeEvaluationSide{
		Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, recipeName),
		Model:      testutil.ArtifactID(t, artifact.KindModel, "attribution-model"),
		Split:      testutil.ArtifactID(t, artifact.KindDatasetShard, "attribution-split"),
		Evaluation: testutil.ArtifactID(t, artifact.KindEvaluation, "attribution-"+recipeName),
		Capability: "long-context retrieval",
		Quality:    0.5, ResourceBytes: 4096, WallNS: 900,
		CoveredCases: 24, TotalCases: 24,
	}
}

// TestRecipeCandidateEvaluationAttribution pins the attribution contract:
// deltas attribute to recipe behavior only when both sides bind the same
// model, split, and capability with complete coverage; a comparison that
// varies the model or split, measures different capabilities, compares one
// recipe against itself, or lacks coverage refuses — an unattributable gain
// is not evidence.
func TestRecipeCandidateEvaluationAttribution(t *testing.T) {
	baseline := attributionSide(t, "baseline-recipe")
	candidate := attributionSide(t, "candidate-recipe")
	candidate.Quality, candidate.ResourceBytes, candidate.WallNS = 0.62, 3072, 700

	attribution, err := AttributeRecipeEvaluation(candidate, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if attribution.QualityDelta <= 0 || attribution.ResourceDelta >= 0 || attribution.WallDelta >= 0 ||
		attribution.Coverage != 24 || attribution.Model != baseline.Model {
		t.Fatalf("controlled attribution = %+v", attribution)
	}

	foreignModel := candidate
	foreignModel.Model = testutil.ArtifactID(t, artifact.KindModel, "other-model")
	if _, err := AttributeRecipeEvaluation(foreignModel, baseline); err == nil ||
		!strings.Contains(err.Error(), "confound weights") {
		t.Fatalf("model-confounded comparison attributed: %v", err)
	}
	foreignSplit := candidate
	foreignSplit.Split = testutil.ArtifactID(t, artifact.KindDatasetShard, "other-split")
	if _, err := AttributeRecipeEvaluation(foreignSplit, baseline); err == nil ||
		!strings.Contains(err.Error(), "confound data") {
		t.Fatalf("split-confounded comparison attributed: %v", err)
	}
	foreignCapability := candidate
	foreignCapability.Capability = "code generation"
	if _, err := AttributeRecipeEvaluation(foreignCapability, baseline); err == nil ||
		!strings.Contains(err.Error(), "not comparable") {
		t.Fatalf("capability-mismatched comparison attributed: %v", err)
	}
	if _, err := AttributeRecipeEvaluation(baseline, baseline); err == nil ||
		!strings.Contains(err.Error(), "nothing varies") {
		t.Fatalf("self-comparison attributed: %v", err)
	}
	uncovered := candidate
	uncovered.CoveredCases = 20
	if _, err := AttributeRecipeEvaluation(uncovered, baseline); err == nil ||
		!strings.Contains(err.Error(), "coverage blocks attribution") {
		t.Fatalf("incomplete coverage attributed: %v", err)
	}
}
