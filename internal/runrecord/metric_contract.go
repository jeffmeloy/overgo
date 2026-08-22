package runrecord

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

// SeedComparison is one seed's parent-versus-child held-out measurement
// (loss: lower is better).
type SeedComparison struct {
	Seed   uint64  `json:"seed"`
	Parent float64 `json:"parent"`
	Child  float64 `json:"child"`
}

// MetricContract is the full descendant-improvement contract one promotion
// must satisfy: identical evaluator and split identities across both
// measurements, multi-seed comparisons, margin beyond the observed parent
// noise envelope, no critical regression, resource cost reported, and the
// holdout query budget respected.
type MetricContract struct {
	Evaluator      artifact.ID      `json:"evaluator"`
	Split          artifact.ID      `json:"split"`
	Comparisons    []SeedComparison `json:"comparisons"`
	CriticalFloor  float64          `json:"critical_floor"`
	CostNS         uint64           `json:"cost_ns"`
	HoldoutQueries uint64           `json:"holdout_queries"`
	HoldoutBudget  uint64           `json:"holdout_budget"`
}

// ValidateDescendantImprovement judges the contract: every seed's child must
// beat its parent; the worst seed improvement must exceed the observed parent
// noise envelope (max-min across seeds -- measured, never a knob); no child
// measurement may cross the critical floor; cost must be reported; and the
// holdout spend must fit its budget. Any violation is a typed refusal.
func ValidateDescendantImprovement(contract MetricContract) error {
	if !contract.Evaluator.Valid() || !contract.Split.Valid() {
		return errors.New("run record: metric contract requires evaluator and split identities")
	}
	if !hasComparisonPair(contract.Comparisons) {
		return fmt.Errorf("run record: metric contract requires multi-seed comparisons, have %d", len(contract.Comparisons))
	}
	if !checked.Nonzero(contract.CostNS) {
		return errors.New("run record: metric contract requires the measured resource cost")
	}
	if !checked.Nonzero(contract.HoldoutBudget) || contract.HoldoutQueries > contract.HoldoutBudget {
		return fmt.Errorf("run record: holdout spend %d exceeds budget %d",
			contract.HoldoutQueries, contract.HoldoutBudget)
	}
	first, _ := checked.First(contract.Comparisons)
	parentLow, parentHigh := first.Parent, first.Parent
	worstImprovement := first.Parent - first.Child
	seen := map[uint64]bool{}
	for _, comparison := range contract.Comparisons {
		if seen[comparison.Seed] {
			return fmt.Errorf("run record: duplicate seed %d in metric contract", comparison.Seed)
		}
		seen[comparison.Seed] = true
		if comparison.Child >= comparison.Parent {
			return fmt.Errorf("run record: seed %d child %.6f does not beat parent %.6f",
				comparison.Seed, comparison.Child, comparison.Parent)
		}
		if checked.PositiveFinite64(contract.CriticalFloor) && comparison.Child > contract.CriticalFloor {
			return fmt.Errorf("run record: seed %d child %.6f crosses the critical floor %.6f",
				comparison.Seed, comparison.Child, contract.CriticalFloor)
		}
		if comparison.Parent < parentLow {
			parentLow = comparison.Parent
		}
		if comparison.Parent > parentHigh {
			parentHigh = comparison.Parent
		}
		worstImprovement = min(worstImprovement, comparison.Parent-comparison.Child)
	}
	noise := parentHigh - parentLow
	if worstImprovement <= noise {
		return fmt.Errorf("run record: worst improvement %.6f does not exceed the observed parent noise envelope %.6f",
			worstImprovement, noise)
	}
	return nil
}

func hasComparisonPair(comparisons []SeedComparison) bool {
	found := false
	for range comparisons {
		if found {
			return true
		}
		found = true
	}
	return false
}
