package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestMoERoutingPromotionRejectsHiddenStratumOrCostRegressions(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	model, dataset, split := id(artifact.KindModel, "model"), id(artifact.KindDataset, "dataset"), id(artifact.KindDatasetShard, "split")
	seed, code, environment := id(artifact.KindEvidence, "seed"), id(artifact.KindEvidence, "code"), id(artifact.KindEvidence, "environment")
	checkpoint, evaluator := id(artifact.KindCheckpoint, "checkpoint"), id(artifact.KindProfile, "evaluator")
	candidate := id(artifact.KindRecipe, "quantile candidate")
	baseline, err := moeRoutingBaselineCodec.New(moeRoutingBaselineMatrix{
		Version: artifact.InitialDocumentVersion,
		moeRoutingPairedAuthorities: moeRoutingPairedAuthorities{
			Model: model, Dataset: dataset, Split: split, DataOrder: id(artifact.KindEvidence, "order"), Seed: seed,
			ComputeBudget: id(artifact.KindEvidence, "budget"), Evaluator: evaluator, Checkpoint: checkpoint,
			Environment: environment, Code: code,
		},
		Arms: []moeRoutingBaselineArm{
			{Kind: moeRoutingNoBalancing, Policy: id(artifact.KindRecipe, "none policy"), Recipe: id(artifact.KindRecipe, "none recipe")},
			{Kind: moeRoutingAuxiliaryLoss, Policy: id(artifact.KindRecipe, "aux policy"), Recipe: id(artifact.KindRecipe, "aux recipe")},
			{Kind: moeRoutingStaticBias, Policy: id(artifact.KindRecipe, "static policy"), Recipe: id(artifact.KindRecipe, "static recipe")},
			{Kind: moeRoutingQuantileBias, Policy: id(artifact.KindRecipe, "quantile policy"), Recipe: candidate},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	segment := func(name string, admitted bool) moeRoutingShiftSegment {
		return moeRoutingShiftSegment{Stratum: id(artifact.KindDatasetShard, name), Evaluation: id(artifact.KindEvidence, name+" evaluation"), Admitted: admitted}
	}
	transitionA, transitionB := segment("transition a", true), segment("transition b", true)
	transitionA.BiasSource, transitionB.BiasSource = artifact.IDPointer(transitionA.Stratum), artifact.IDPointer(transitionB.Stratum)
	shift, err := moeRoutingShiftCodec.New(moeRoutingShiftEvidence{
		Version: artifact.InitialDocumentVersion, BaselineMatrix: baseline.ID,
		Cases: []moeRoutingShiftCase{
			{Kind: moeRoutingHomogeneous, AggregateEvaluation: id(artifact.KindEvidence, "homogeneous"), AggregateAdmitted: true, Segments: []moeRoutingShiftSegment{segment("homogeneous segment", true)}},
			{Kind: moeRoutingMixed, AggregateEvaluation: id(artifact.KindEvidence, "mixed"), AggregateAdmitted: true, Segments: []moeRoutingShiftSegment{segment("mixed a", true), segment("mixed b", true)}},
			{Kind: moeRoutingImbalanced, AggregateEvaluation: id(artifact.KindEvidence, "imbalanced"), AggregateAdmitted: true, Segments: []moeRoutingShiftSegment{segment("imbalanced a", true), segment("imbalanced b", true)}},
			{Kind: moeRoutingTransition, AggregateEvaluation: id(artifact.KindEvidence, "transition"), AggregateAdmitted: true, Segments: []moeRoutingShiftSegment{transitionA, transitionB}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	capacity, err := moeCapacityPairCodec.New(moeCapacityPair{
		Version: artifact.InitialDocumentVersion, Model: model, Dataset: dataset, Split: split, Checkpoint: checkpoint,
		Seed: seed, Code: code, Environment: environment, Evaluator: evaluator,
		Constrained: moeCapacityEndpoint{Run: id(artifact.KindRun, "constrained"), Policy: candidate, Evaluation: id(artifact.KindEvidence, "constrained evaluation"), Dropped: 1, Total: 10, Admitted: true},
		Dropless:    moeCapacityEndpoint{Run: id(artifact.KindRun, "dropless"), Policy: id(artifact.KindRecipe, "dropless policy"), Evaluation: id(artifact.KindEvidence, "dropless evaluation"), Total: 10, Admitted: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	policy, err := moeRoutingPromotionPolicyCodec.New(moeRoutingPromotionPolicy{
		Version:          artifact.InitialDocumentVersion,
		QualityEvaluator: id(artifact.KindProfile, "quality evaluator"), StabilityEvaluator: id(artifact.KindProfile, "stability evaluator"),
		UtilizationEvaluator: id(artifact.KindProfile, "utilization evaluator"), DropEvaluator: id(artifact.KindProfile, "drop evaluator"),
		CostEvaluator: id(artifact.KindProfile, "cost evaluator"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(moeRoutingPromotionPolicyLineage(policy)) != 5 {
		t.Fatal("promotion policy omits an evaluator authority")
	}
	comparison := func(evaluator artifact.ID, name string, outcome moeRoutingComparisonOutcome) moeRoutingPromotionComparison {
		return moeRoutingPromotionComparison{Evaluator: evaluator, Evidence: id(artifact.KindEvidence, name), Outcome: outcome}
	}
	comparisons := moeRoutingPromotionComparisons{
		Quality:     comparison(policy.QualityEvaluator, "quality", moeRoutingImproved),
		Stability:   comparison(policy.StabilityEvaluator, "stability", moeRoutingEquivalent),
		Utilization: comparison(policy.UtilizationEvaluator, "utilization", moeRoutingEquivalent),
		Drop:        comparison(policy.DropEvaluator, "drop", moeRoutingEquivalent),
		Cost:        comparison(policy.CostEvaluator, "cost", moeRoutingEquivalent),
	}
	promotion, err := buildMoERoutingPromotion(policy, baseline, shift, capacity, candidate, comparisons)
	if err != nil || len(moeRoutingPromotionLineage(promotion)) != 10 {
		t.Fatalf("promotion = (%+v, %v)", promotion, err)
	}
	hidden := shift
	hidden.ID, hidden.Cases = artifact.ID{}, cloneMoERoutingShiftCases(shift.Cases)
	for index := range hidden.Cases {
		if hidden.Cases[index].Kind == moeRoutingMixed {
			hidden.Cases[index].Segments[0].Admitted = false
		}
	}
	hidden.HiddenStratumFailure = true
	hidden, err = moeRoutingShiftCodec.New(hidden)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := buildMoERoutingPromotion(policy, baseline, hidden, capacity, candidate, comparisons); err == nil {
		t.Fatal("hidden stratum regression promoted")
	}
	costRegression := comparisons
	costRegression.Cost.Outcome = moeRoutingComparisonOutcome("regressed")
	if _, err := buildMoERoutingPromotion(policy, baseline, shift, capacity, candidate, costRegression); err == nil {
		t.Fatal("cost regression promoted")
	}
}
