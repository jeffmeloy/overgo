package evaluation

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestEffectiveSpeedupUsesObservedQualityAndCompatibleCostUnits(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	evaluator, regime := id(artifact.KindProfile, "quality evaluator"), id(artifact.KindProfile, "regime")
	makeFit := func(name string, costs []string) scalingFit {
		point := func(rung uint32, quality, cost string) scalingFitPoint {
			return scalingFitPoint{
				Rung: rung, Run: id(artifact.KindRun, name+" run "+cost),
				Accounting: id(artifact.KindEvidence, name+" accounting "+cost),
				Evaluation: id(artifact.KindEvidence, name+" evaluation "+cost), Regime: regime, CostDecimal: cost,
				Observed:  scalingRational{Numerator: quality, Denominator: "1"},
				Predicted: scalingRational{Numerator: quality, Denominator: "1"},
				Residual:  scalingRational{Numerator: "0", Denominator: "1"},
				Uncertainty: scalingUncertaintyEnvelope{
					Lower: scalingRational{Numerator: quality, Denominator: "1"},
					Upper: scalingRational{Numerator: quality, Denominator: "1"},
				}, Covered: true,
			}
		}
		fit, err := scalingFitCodec.New(scalingFit{
			Version: artifact.InitialDocumentVersion, Study: id(artifact.KindEvidence, name+" study"),
			CostAxis: scalingCostMeasuredFLOPs, QualityEvaluator: evaluator, FitEvaluator: id(artifact.KindProfile, name+" fit evaluator"),
			FitMethod: id(artifact.KindProfile, name+" fit method"), UncertaintyMethod: id(artifact.KindProfile, name+" uncertainty method"),
			UncertaintyEvidence: id(artifact.KindEvidence, name+" uncertainty"), Regime: regime,
			RegimeEvidence: id(artifact.KindEvidence, name+" regime evidence"), RegimeReopenTrigger: id(artifact.KindRecipe, name+" regime reopen"),
			ObservedMin: costs[0], ObservedMax: costs[2], ClaimMin: costs[0], ClaimMax: costs[2],
			ResidualEvaluation: id(artifact.KindEvidence, name+" residual evaluation"), ResidualAdmitted: true,
			HoldoutEvaluation: id(artifact.KindEvidence, name+" holdout evaluation"), HoldoutAdmitted: true,
			FitPoints: []scalingFitPoint{point(0, "10", costs[0]), point(2, "30", costs[2])},
			Holdouts:  []scalingFitPoint{point(1, "20", costs[1])},
			Assumptions: []scalingFitAssumption{{
				Name: "observed points only", Evidence: id(artifact.KindEvidence, name+" assumption"),
				ReopenTrigger: id(artifact.KindRecipe, name+" assumption reopen"),
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		return fit
	}
	baseline, candidate := makeFit("baseline", []string{"100", "200", "300"}), makeFit("candidate", []string{"50", "100", "150"})
	evidence, err := buildEffectiveSpeedup(baseline, candidate)
	if err != nil || len(evidence.Pairs) != 3 {
		t.Fatalf("effective speedup = (%+v, %v)", evidence, err)
	}
	for _, pair := range evidence.Pairs {
		if pair.Speedup != (scalingRational{Numerator: "2", Denominator: "1"}) {
			t.Fatalf("speedup pair = %+v", pair)
		}
	}
	content, err := effectiveSpeedupCodec.Content(evidence)
	if err != nil || content.Descriptor.ID != evidence.ID || len(effectiveSpeedupLineage(evidence)) != 22 {
		t.Fatalf("effective speedup content = (%+v, %v)", content, err)
	}
	wrongUnits := candidate
	wrongUnits.ID, wrongUnits.CostAxis = artifact.ID{}, scalingCostWallNS
	wrongUnits, err = scalingFitCodec.New(wrongUnits)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := buildEffectiveSpeedup(baseline, wrongUnits); err == nil {
		t.Fatal("incompatible cost units admitted")
	}
	noQualityMatch := candidate
	noQualityMatch.ID, noQualityMatch.FitPoints, noQualityMatch.Holdouts = artifact.ID{}, slicesCloneScalingFitPoints(candidate.FitPoints), slicesCloneScalingFitPoints(candidate.Holdouts)
	for index := range noQualityMatch.FitPoints {
		noQualityMatch.FitPoints[index].Observed.Numerator = "99"
		noQualityMatch.FitPoints[index].Predicted.Numerator = "99"
		noQualityMatch.FitPoints[index].Uncertainty.Lower.Numerator = "99"
		noQualityMatch.FitPoints[index].Uncertainty.Upper.Numerator = "99"
	}
	noQualityMatch.Holdouts[0].Observed.Numerator, noQualityMatch.Holdouts[0].Predicted.Numerator = "99", "99"
	noQualityMatch.Holdouts[0].Uncertainty.Lower.Numerator, noQualityMatch.Holdouts[0].Uncertainty.Upper.Numerator = "99", "99"
	noQualityMatch, err = scalingFitCodec.New(noQualityMatch)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := buildEffectiveSpeedup(baseline, noQualityMatch); err == nil {
		t.Fatal("predicted or unmatched quality produced speedup")
	}
}

func TestEffectiveSpeedupCanonicalOrderIsPermutationInvariant(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	baseline := effectiveSpeedupEndpoint{
		Rung: 1, Run: id(artifact.KindRun, "shared baseline run"),
		Accounting: id(artifact.KindEvidence, "shared baseline accounting"),
		Evaluation: id(artifact.KindEvidence, "shared baseline evaluation"), CostDecimal: "100",
	}
	pair := func(name, cost, numerator string) effectiveSpeedupPair {
		return effectiveSpeedupPair{
			Quality: scalingRational{Numerator: "10", Denominator: "1"}, Baseline: baseline,
			Candidate: effectiveSpeedupEndpoint{
				Rung: 1, Run: id(artifact.KindRun, name+" run"),
				Accounting: id(artifact.KindEvidence, name+" accounting"),
				Evaluation: id(artifact.KindEvidence, name+" evaluation"), CostDecimal: cost,
			},
			Speedup: scalingRational{Numerator: numerator, Denominator: "1"},
		}
	}
	pairs := []effectiveSpeedupPair{pair("quarter", "25", "4"), pair("half", "50", "2"), pair("equal", "100", "1")}
	base := effectiveSpeedup{
		Version:     artifact.InitialDocumentVersion,
		BaselineFit: id(artifact.KindEvidence, "baseline fit"), CandidateFit: id(artifact.KindEvidence, "candidate fit"),
		CostAxis: scalingCostMeasuredFLOPs, QualityEvaluator: id(artifact.KindProfile, "quality evaluator"),
		Regime: id(artifact.KindProfile, "regime"),
	}
	permutations := [][]effectiveSpeedupPair{
		slices.Clone(pairs),
		{pairs[2], pairs[0], pairs[1]},
		{pairs[1], pairs[2], pairs[0]},
	}
	var canonical effectiveSpeedup
	for index, permutation := range permutations {
		candidate := base
		candidate.Pairs = permutation
		identified, err := effectiveSpeedupCodec.New(candidate)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.IsSortedFunc(identified.Pairs, compareEffectiveSpeedupPairs) {
			t.Fatal("effective speedup pairs are not in canonical total order")
		}
		if index == 0 {
			canonical = identified
			continue
		}
		if identified.ID != canonical.ID || !slices.Equal(identified.Pairs, canonical.Pairs) {
			t.Fatalf("permutation %d changed canonical evidence: %s != %s", index, identified.ID, canonical.ID)
		}
	}
}
