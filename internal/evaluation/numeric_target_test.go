package evaluation

import (
	"testing"

	"overgo/internal/recipecontract"
)

func TestNumericTargetEvaluationAvoidsShapeAndDistributionAssumptions(t *testing.T) {
	quantile := 0.8
	for _, modality := range []recipecontract.Modality{recipecontract.ModalityTimeSeries, recipecontract.ModalityTable} {
		view := textTargetView(t, modality)
		view.Signature.Outputs[0] = modality
		view, err := sftEvaluationViewCodec.New(view)
		if err != nil {
			t.Fatal(err)
		}
		plan, err := CompileNumericTargetPlan(view, NumericTargetSuite{
			Scorers: []NumericScorerSpec{
				{Kind: NumericAbsoluteError, Unit: "declared-unit"},
				{Kind: NumericQuantileLoss, Unit: "declared-unit", Quantile: &quantile},
				{Kind: NumericLabelAccuracy},
			},
			Cases: []NumericTargetCase{{
				Record: "heldout", Values: []float64{-1000, 2, 3.5}, Label: "class-b",
				Assumptions: []NumericAssumption{{Name: "target-order", Value: "dataset-declared"}},
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		report, err := ScoreNumericTargets(plan, []NumericTargetObservation{{
			Record: "heldout", Values: []float64{-999, 4, -7}, Label: "class-b",
		}})
		if err != nil || len(report.Records) != 1 || len(report.Metrics) != len(plan.Scorers) ||
			len(report.Records[0].Expected) != 3 || len(report.Records[0].Assumptions) != 1 {
			t.Fatalf("%s report=%+v err=%v", modality, report, err)
		}
	}
}
