package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestSplitIsolationAndQueryBudget pins the evaluation-isolation contract:
// the four split roles bind to distinct shards, query budgets are immutable
// grants whose balance derives from committed charges and can prove exhaustion
// but never go silently negative, and the partition's proposer is blinded from
// promotion and audit results while other observers and roles stay open.
func TestSplitIsolationAndQueryBudget(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	proposer := id(artifact.KindEvidence, "proposer")
	evaluator := id(artifact.KindEvidence, "evaluator")
	partition := testcheck.Must(splitPartitionCodec.New(SplitPartition{
		Version:     artifact.InitialDocumentVersion,
		Dataset:     id(artifact.KindDataset, "dataset"),
		Development: id(artifact.KindDatasetShard, "development"),
		Selection:   id(artifact.KindDatasetShard, "selection"),
		Promotion:   id(artifact.KindDatasetShard, "promotion"),
		Audit:       id(artifact.KindDatasetShard, "audit"),
		Proposer:    proposer,
	}))(t)

	duplicate := partition
	duplicate.ID = artifact.ID{}
	duplicate.Audit = duplicate.Promotion
	if _, err := splitPartitionCodec.New(duplicate); err == nil {
		t.Fatal("partition with shared promotion/audit shard accepted")
	}

	budget := testcheck.Must(budgetCodec.New(Budget{
		Version: artifact.InitialDocumentVersion, Unit: "queries", Split: partition.Promotion,
		Issued: 5, Authority: id(artifact.KindEvidence, "authority"),
	}))(t)
	charge := func(amount uint64, purpose string) BudgetCharge {
		value := testcheck.Must(budgetChargeCodec.New(BudgetCharge{
			Version: artifact.InitialDocumentVersion, Budget: budget.ID, Amount: amount,
			Consumer: evaluator, Purpose: purpose,
		}))(t)
		return value
	}
	remaining, err := BudgetBalance(budget, []BudgetCharge{charge(2, "selection eval"), charge(3, "promotion eval")})
	if err != nil || remaining != 0 {
		t.Fatalf("balance = (%d, %v), want exhausted grant", remaining, err)
	}
	if _, err := BudgetBalance(budget, []BudgetCharge{charge(4, "first"), charge(2, "overrun")}); err == nil ||
		!strings.Contains(err.Error(), "exhausted") {
		t.Fatalf("over-consumption not refused: %v", err)
	}
	foreign := charge(1, "foreign")
	foreign.Budget = id(artifact.KindEvidence, "other-budget")
	foreign.ID = artifact.ID{}
	foreign = testcheck.Must(budgetChargeCodec.New(foreign))(t)
	if _, err := BudgetBalance(budget, []BudgetCharge{foreign}); err == nil {
		t.Fatal("charge against another budget accepted")
	}
	roundTrip, err := ParseBudget(mustContent(t, budget))
	if err != nil || roundTrip.ID != budget.ID || roundTrip.Issued != 5 {
		t.Fatalf("budget round trip = (%+v, %v)", roundTrip, err)
	}

	proposerCharge := charge(1, "proposer query")
	proposerCharge.Consumer = proposer
	proposerCharge.ID = artifact.ID{}
	proposerCharge = testcheck.Must(budgetChargeCodec.New(proposerCharge))(t)
	blindedBudget := func(split artifact.ID, name string) Budget {
		value := testcheck.Must(budgetCodec.New(Budget{
			Version: artifact.InitialDocumentVersion, Unit: "queries", Split: split,
			Issued: 5, Authority: id(artifact.KindEvidence, "authority-"+name),
		}))(t)
		return value
	}
	if err := ValidateBlinding(partition, blindedBudget(partition.Promotion, "promotion"), []BudgetCharge{proposerCharge}); err == nil {
		t.Fatal("proposer observed promotion results")
	}
	if err := ValidateBlinding(partition, blindedBudget(partition.Audit, "audit"), []BudgetCharge{proposerCharge}); err == nil {
		t.Fatal("proposer observed audit results")
	}
	if err := ValidateBlinding(partition, blindedBudget(partition.Development, "development"), []BudgetCharge{proposerCharge}); err != nil {
		t.Fatalf("proposer blocked from development: %v", err)
	}
	if err := ValidateBlinding(partition, blindedBudget(partition.Selection, "selection"), []BudgetCharge{proposerCharge}); err != nil {
		t.Fatalf("proposer blocked from selection: %v", err)
	}
	if err := ValidateBlinding(partition, blindedBudget(partition.Promotion, "promotion"), []BudgetCharge{charge(1, "evaluator query")}); err != nil {
		t.Fatalf("evaluator blocked from promotion: %v", err)
	}
	if err := ValidateBlinding(partition, blindedBudget(id(artifact.KindDatasetShard, "foreign-split"), "foreign"), []BudgetCharge{proposerCharge}); err != nil {
		t.Fatalf("ungoverned split refused: %v", err)
	}
}

func mustContent(t *testing.T, budget Budget) []byte {
	t.Helper()
	content := testcheck.Must(budgetCodec.Content(budget))(t)
	return content.Data
}
