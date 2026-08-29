package evaluation

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestScalingFitRefusesUnsupportedExtrapolationAndRegimeMixing(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	rational := func(numerator string) scalingRational { return scalingRational{Numerator: numerator, Denominator: "1"} }
	regime := id(artifact.KindProfile, "regime")
	point := func(rung uint32, cost, observed, predicted, residual, lower, upper string) scalingFitPoint {
		return scalingFitPoint{
			Rung: rung, Run: id(artifact.KindRun, "run "+cost), Accounting: id(artifact.KindEvidence, "accounting "+cost),
			Evaluation: id(artifact.KindEvidence, "evaluation "+cost), Regime: regime, CostDecimal: cost,
			Observed: rational(observed), Predicted: rational(predicted), Residual: rational(residual),
			Uncertainty: scalingUncertaintyEnvelope{Lower: rational(lower), Upper: rational(upper)}, Covered: true,
		}
	}
	fit, err := scalingFitCodec.New(scalingFit{
		Version: artifact.InitialDocumentVersion, Study: id(artifact.KindEvidence, "study"),
		CostAxis: scalingCostActiveParameters, QualityEvaluator: id(artifact.KindProfile, "quality evaluator"),
		FitEvaluator: id(artifact.KindProfile, "fit evaluator"),
		FitMethod:    id(artifact.KindProfile, "fit method"), UncertaintyMethod: id(artifact.KindProfile, "uncertainty method"),
		UncertaintyEvidence: id(artifact.KindEvidence, "uncertainty evidence"), Regime: regime,
		RegimeEvidence: id(artifact.KindEvidence, "regime evidence"), RegimeReopenTrigger: id(artifact.KindRecipe, "regime reopen"),
		ObservedMin: "100", ObservedMax: "300", ClaimMin: "100", ClaimMax: "300",
		ResidualEvaluation: id(artifact.KindEvidence, "residual evaluation"), ResidualAdmitted: true,
		HoldoutEvaluation: id(artifact.KindEvidence, "holdout evaluation"), HoldoutAdmitted: true,
		FitPoints: []scalingFitPoint{point(2, "300", "30", "28", "2", "27", "31"), point(0, "100", "10", "11", "-1", "9", "12")},
		Holdouts:  []scalingFitPoint{point(1, "200", "20", "19", "1", "18", "21")},
		Assumptions: []scalingFitAssumption{{
			Name: "single observed regime", Evidence: id(artifact.KindEvidence, "assumption evidence"),
			ReopenTrigger: id(artifact.KindRecipe, "assumption reopen"),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := scalingFitCodec.Content(fit)
	if err != nil || content.Descriptor.ID != fit.ID || fit.FitPoints[0].CostDecimal != "100" || len(scalingFitLineage(fit)) != 22 {
		t.Fatalf("scaling fit = (%+v, %v)", fit, err)
	}
	extrapolated := fit
	extrapolated.ID, extrapolated.ClaimMax = artifact.ID{}, "301"
	if _, err := scalingFitCodec.New(extrapolated); err == nil {
		t.Fatal("unsupported extrapolation admitted")
	}
	mixed := fit
	mixed.ID, mixed.Holdouts = artifact.ID{}, slicesCloneScalingFitPoints(fit.Holdouts)
	mixed.Holdouts[0].Regime = id(artifact.KindProfile, "other regime")
	if _, err := scalingFitCodec.New(mixed); err == nil {
		t.Fatal("mixed-regime fit admitted")
	}
	wrongResidual := fit
	wrongResidual.ID, wrongResidual.FitPoints = artifact.ID{}, slicesCloneScalingFitPoints(fit.FitPoints)
	wrongResidual.FitPoints[0].Residual = rational("0")
	if _, err := scalingFitCodec.New(wrongResidual); err == nil {
		t.Fatal("incorrect residual admitted")
	}
	advisory := fit
	advisory.ID, advisory.HoldoutAdmitted, advisory.AdvisoryOnly = artifact.ID{}, false, true
	if _, err := scalingFitCodec.New(advisory); err != nil {
		t.Fatalf("falsified holdout was not retained as advisory evidence: %v", err)
	}
}

func slicesCloneScalingFitPoints(points []scalingFitPoint) []scalingFitPoint {
	return append([]scalingFitPoint(nil), points...)
}
