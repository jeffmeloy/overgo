package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestDescendantImprovementRequiresEnvelopeSeparationForEverySeed(t *testing.T) {
	contract := MetricContract{
		Evaluator: testutil.ArtifactID(t, artifact.KindEvidence, "metric-contract-evaluator"),
		Split:     testutil.ArtifactID(t, artifact.KindDataset, "metric-contract-split"),
		Comparisons: []SeedComparison{
			{Seed: 1, Parent: 10, Child: 9},
			{Seed: 2, Parent: 12, Child: 8},
		},
		CostNS: 1, HoldoutQueries: 1, HoldoutBudget: 1,
	}
	if err := ValidateDescendantImprovement(contract); err == nil {
		t.Fatal("mean-only win admitted a seed inside the observed parent envelope")
	}
	contract.Comparisons[0].Child = 7
	contract.Comparisons[1].Child = 9
	if err := ValidateDescendantImprovement(contract); err != nil {
		t.Fatalf("every-seed envelope separation was refused: %v", err)
	}
}
