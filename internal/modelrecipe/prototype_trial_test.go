package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func prototypeTrialInputs(t *testing.T) (ModelPrototype, artifact.ID, trainingprogram.ScratchSpec, artifact.ID) {
	t.Helper()
	prototype, err := NewModelPrototype(prototypeFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	admission := testutil.ArtifactID(t, artifact.KindEvidence, "trial-admission")
	construction := trainingprogram.ScratchSpec{
		Recipe:             testutil.ArtifactID(t, artifact.KindRecipe, "trial-construction-recipe"),
		Dataset:            prototype.DevelopmentSplit,
		Split:              testutil.ArtifactID(t, artifact.KindDatasetShard, "trial-shard"),
		DerivationProfile:  prototype.DerivationPolicy,
		TopologyProfile:    testutil.ArtifactID(t, artifact.KindProfile, "trial-topology"),
		Tokenizer:          testutil.ArtifactID(t, artifact.KindTokenizer, "trial-tokenizer"),
		ParameterManifest:  testutil.ArtifactID(t, artifact.KindTensorInventory, "trial-parameters"),
		InitializerProfile: testutil.ArtifactID(t, artifact.KindProfile, "trial-initializer"),
		InitializedModel:   testutil.ArtifactID(t, artifact.KindModel, "trial-initialized"),
	}
	algorithm := testutil.ArtifactID(t, artifact.KindProfile, "trial-rng-algorithm")
	for seed, name := range map[uint64]string{11: "split", 13: "init", 17: "data", 19: "augmentation"} {
		construction.RNGStreams = append(construction.RNGStreams, trainingprogram.RNGStreamSpec{
			Name: name, Algorithm: algorithm, Seed: seed,
		})
	}
	evaluationPlan := testutil.ArtifactID(t, artifact.KindEvidence, "trial-evaluation-plan")
	return prototype, admission, construction, evaluationPlan
}

// TestPrototypeCompilesClosedWorldTrial pins the trial compiler: an admitted
// prototype compiles into scratch-construction authority, a registered
// architecture's tensor-shape draft, a supported objective, the bound
// evaluation plan, and deduplicated recipe dependencies — all identity-level
// derivation through existing owners with no source generation — while an
// unsupported objective, a foreign development split, a foreign derivation
// policy, or an unbound evaluation plan refuses before any compute.
func TestPrototypeCompilesClosedWorldTrial(t *testing.T) {
	prototype, admission, construction, evaluationPlan := prototypeTrialInputs(t)
	trial, err := CompilePrototypeTrial(
		prototype, admission, construction, trainingprogram.ObjectiveTokenPrediction, evaluationPlan, 8,
	)
	if err != nil {
		t.Fatal(err)
	}
	if trial.Architecture.Name != prototype.Architecture || trial.Draft.Heads != 8 {
		t.Fatalf("trial architecture draft = %+v", trial.Draft)
	}
	if trial.Construction.ID().Kind() != artifact.KindRecipe {
		t.Fatalf("trial construction authority = %+v", trial.Construction.ID())
	}
	if len(trial.CandidateRecipes) != 2 || trial.EvaluationPlan != evaluationPlan {
		t.Fatalf("trial dependencies = %+v", trial)
	}

	if _, err := CompilePrototypeTrial(
		prototype, admission, construction, trainingprogram.ObjectiveKind("weight-surgery"), evaluationPlan, 8,
	); err == nil || !strings.Contains(err.Error(), "refused before compute") {
		t.Fatalf("unsupported objective compiled: %v", err)
	}
	foreignSplit := construction
	foreignSplit.Dataset = testutil.ArtifactID(t, artifact.KindDataset, "foreign-split")
	if _, err := CompilePrototypeTrial(
		prototype, admission, foreignSplit, trainingprogram.ObjectiveTokenPrediction, evaluationPlan, 8,
	); err == nil || !strings.Contains(err.Error(), "development split") {
		t.Fatalf("foreign development split compiled: %v", err)
	}
	foreignPolicy := construction
	foreignPolicy.DerivationProfile = testutil.ArtifactID(t, artifact.KindProfile, "foreign-derivation")
	if _, err := CompilePrototypeTrial(
		prototype, admission, foreignPolicy, trainingprogram.ObjectiveTokenPrediction, evaluationPlan, 8,
	); err == nil || !strings.Contains(err.Error(), "derivation policy") {
		t.Fatalf("foreign derivation policy compiled: %v", err)
	}
	if _, err := CompilePrototypeTrial(
		prototype, admission, construction, trainingprogram.ObjectiveTokenPrediction, artifact.ID{}, 8,
	); err == nil || !strings.Contains(err.Error(), "evaluation plan") {
		t.Fatalf("unbound evaluation plan compiled: %v", err)
	}
	if _, err := CompilePrototypeTrial(
		prototype, admission, construction, trainingprogram.ObjectiveTokenPrediction, evaluationPlan, 0,
	); err == nil || !strings.Contains(err.Error(), "head count") {
		t.Fatalf("zero-head tensor draft compiled: %v", err)
	}
}
