package evaluation

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestCompositeGenerationHeldoutQuality(t *testing.T) {
	store, evidenceID, policy := compositeGenerationQualityFixture(t)
	authority := CompositeGenerationQualityAuthority{}
	quality, err := authority.Evaluate(context.Background(), store, evidenceID, policy)
	if err != nil {
		t.Fatal(err)
	}
	if quality.ID.Kind() != artifact.KindEvidence || quality.GenerationEvidence != evidenceID ||
		quality.WorstComposedQuality != 0.88 || quality.WorstHeldOutGain < policy.MinimumHeldOutGain ||
		quality.WorstNondegeneracy < policy.MinimumNondegeneracy {
		t.Fatalf("quality = %+v", quality)
	}
	content, err := quality.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := authority.Parse(content.Data)
	if err != nil || parsed.ID != quality.ID || parsed.ValidateIdentity() != nil {
		t.Fatalf("quality round trip = %+v, %v", parsed, err)
	}
}

func TestCompositeGenerationRepeatedSeeds(t *testing.T) {
	store, evidenceID, policy := compositeGenerationQualityFixture(t)
	authority := CompositeGenerationQualityAuthority{}
	policy.MinimumSeeds++
	policy, err := authority.NewPolicy(policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := authority.Evaluate(context.Background(), store, evidenceID, policy); err == nil {
		t.Fatal("insufficient repeated seeds admitted")
	}
	_, _, policy = compositeGenerationQualityFixture(t)
	policy.MaximumSeedSpread = 0.01
	policy, err = authority.NewPolicy(policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := authority.Evaluate(context.Background(), store, evidenceID, policy); err == nil {
		t.Fatal("unstable repeated seeds admitted")
	}
}

func TestCompositeGenerationSourceEffect(t *testing.T) {
	store, evidenceID, policy := compositeGenerationQualityFixture(t)
	policy.MinimumSourceEffect = 0.5
	policy, err := (CompositeGenerationQualityAuthority{}).NewPolicy(policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (CompositeGenerationQualityAuthority{}).Evaluate(
		context.Background(), store, evidenceID, policy,
	); err == nil {
		t.Fatal("composition without attributable source effect admitted")
	}
}

func TestCompositeGenerationRefusal(t *testing.T) {
	store, evidenceID, policy := compositeGenerationQualityFixture(t)
	policy.QualityMetric = "missing-quality"
	policy, err := (CompositeGenerationQualityAuthority{}).NewPolicy(policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (CompositeGenerationQualityAuthority{}).Evaluate(
		context.Background(), store, evidenceID, policy,
	); err == nil {
		t.Fatal("missing held-out metric admitted")
	}
	_, _, policy = compositeGenerationQualityFixture(t)
	policy.MinimumNondegeneracy = 0.95
	policy, err = (CompositeGenerationQualityAuthority{}).NewPolicy(policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (CompositeGenerationQualityAuthority{}).Evaluate(
		context.Background(), store, evidenceID, policy,
	); err == nil {
		t.Fatal("degenerate generation admitted")
	}
}

func compositeGenerationQualityFixture(
	t *testing.T,
) (*overgodb.Store, artifact.ID, CompositeGenerationQualityPolicy) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	ctx := context.Background()
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	source := artifact.Descriptor{ID: id(artifact.KindModel, "quality source model"), Size: 11}
	target := artifact.Descriptor{ID: id(artifact.KindModel, "quality target model"), Size: 13}
	bridge := artifact.Descriptor{ID: id(artifact.KindAdapter, "quality bridge"), Size: 17}
	recipes := map[composition.CompositeGenerationArm]artifact.ID{
		composition.CompositeGenerationTargetBaseline: id(artifact.KindRecipe, "quality target baseline"),
		composition.CompositeGenerationComposed:       id(artifact.KindRecipe, "quality composed"),
		composition.CompositeGenerationSourceAblated:  id(artifact.KindRecipe, "quality source ablated"),
		composition.CompositeGenerationBridgeAblated:  id(artifact.KindRecipe, "quality bridge ablated"),
	}
	dataset := id(artifact.KindDataset, "quality heldout dataset")
	split := id(artifact.KindDatasetShard, "quality heldout split")
	evaluator := id(artifact.KindEvidence, "quality evaluator")
	environment := id(artifact.KindEvidence, "quality environment")
	plan := id(artifact.KindProfile, "quality execution plan")
	descriptors := []artifact.Descriptor{source, target, bridge, {ID: dataset}, {ID: split}, {ID: evaluator}, {ID: environment}, {ID: plan}}
	for _, recipeID := range recipes {
		descriptors = append(descriptors, artifact.Descriptor{ID: recipeID})
	}
	outputContract := artifact.DocumentContract{
		Kind: artifact.KindOutput, MediaType: "application/vnd.overgo.quality-fixture+json", Schema: "overgo/quality-fixture/v1",
	}
	inputContent, err := outputContract.ContentBytes([]byte(`{"heldout":"input"}`))
	if err != nil {
		t.Fatal(err)
	}
	contents := []artifact.Content{inputContent}
	lineage := []artifact.Lineage{}
	trials := []composition.CompositeGenerationTrial{}
	for _, seed := range []uint64{19, 23} {
		for _, arm := range []composition.CompositeGenerationArm{
			composition.CompositeGenerationTargetBaseline,
			composition.CompositeGenerationComposed,
			composition.CompositeGenerationSourceAblated,
			composition.CompositeGenerationBridgeAblated,
		} {
			outputContent, contentErr := outputContract.ContentBytes([]byte(fmt.Sprintf(`{"arm":%q,"seed":%d}`, arm, seed)))
			if contentErr != nil {
				t.Fatal(contentErr)
			}
			run, runErr := runrecord.NewBoundRun(
				recipes[arm], runrecord.OutcomeSucceeded, []artifact.ID{inputContent.Descriptor.ID},
				[]artifact.ID{outputContent.Descriptor.ID}, "", strings.Repeat("b", 40), environment, 100, nil,
			)
			if runErr != nil {
				t.Fatal(runErr)
			}
			quality := compositeGenerationFixtureQuality(arm, seed)
			evaluation, evaluationErr := runrecord.NewEvaluation(recipes[arm], run.ID, dataset, []runrecord.Metric{
				{Name: "quality-score", Value: quality, Direction: runrecord.DirectionMaximize},
				{Name: "nondegeneracy-score", Value: 0.9, Direction: runrecord.DirectionMaximize},
			})
			if evaluationErr != nil {
				t.Fatal(evaluationErr)
			}
			runContent, _ := run.Content()
			evaluationContent, _ := evaluation.Content()
			contents = append(contents, outputContent, runContent, evaluationContent)
			lineage = append(lineage, run.Lineage()...)
			lineage = append(lineage, evaluation.Lineage()...)
			observation := id(artifact.KindEvidence, fmt.Sprintf("quality observation %s %d", arm, seed))
			descriptors = append(descriptors, artifact.Descriptor{ID: observation})
			trials = append(trials, composition.CompositeGenerationTrial{
				Seed: seed, Arm: arm, Input: inputContent.Descriptor.ID, Output: outputContent.Descriptor.ID,
				Run: run.ID, Evaluation: evaluation.ID, Observation: observation,
			})
		}
	}
	evidence, err := (composition.CompositeGenerationEvidenceAuthority{}).New(composition.CompositeGenerationEvidence{
		SourceModel: composition.CompositeGenerationFrozenModel{Before: source, After: source},
		TargetModel: composition.CompositeGenerationFrozenModel{Before: target, After: target},
		Bridge:      bridge, ExecutionPlan: plan,
		TargetBaselineRecipe: recipes[composition.CompositeGenerationTargetBaseline],
		CompositionRecipe:    recipes[composition.CompositeGenerationComposed],
		SourceAblatedRecipe:  recipes[composition.CompositeGenerationSourceAblated],
		BridgeAblatedRecipe:  recipes[composition.CompositeGenerationBridgeAblated],
		Dataset:              dataset, HeldOutSplit: split, Evaluator: evaluator, Trials: trials,
	})
	if err != nil {
		t.Fatal(err)
	}
	evidenceContent, err := evidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	contents = append(contents, evidenceContent)
	lineage = append(lineage, evidence.Lineage()...)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/composite-generation/quality", Artifacts: descriptors, Contents: contents, Lineage: lineage,
	}); err != nil {
		t.Fatal(err)
	}
	policy, err := (CompositeGenerationQualityAuthority{}).NewPolicy(CompositeGenerationQualityPolicy{
		QualityMetric: "quality-score", NondegeneracyMetric: "nondegeneracy-score",
		Direction: runrecord.DirectionMaximize, MinimumSeeds: 2, ComposedQualityThreshold: 0.8,
		MinimumHeldOutGain: 0.1, MinimumSourceEffect: 0.2, MinimumBridgeEffect: 0.25,
		MaximumSeedSpread: 0.05, MinimumNondegeneracy: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}
	return store, evidence.ID, policy
}

func compositeGenerationFixtureQuality(arm composition.CompositeGenerationArm, seed uint64) float64 {
	switch arm {
	case composition.CompositeGenerationComposed:
		if seed == 19 {
			return 0.90
		}
		return 0.88
	case composition.CompositeGenerationTargetBaseline:
		return 0.70
	case composition.CompositeGenerationSourceAblated:
		return 0.60
	default:
		return 0.55
	}
}
