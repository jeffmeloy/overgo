package gate

import (
	"slices"
	"testing"
)

// batchPromotionControl decides whether batched acceptance may be promoted from
// the sequential floor. Promotion is admitted only after all three preconditions
// the campaign requires hold: crash recovery is proven (a killed producer or
// supervisor resumes its committed receipts), change selection is safe (only the
// affected packages re-run), and the batch shows measured payback (reuse that
// saved re-execution wall, not merely fewer commits). A missing precondition
// keeps the sequential floor and names why, so batching is never promoted on an
// unproven or unmeasured basis.
func batchPromotionControl(recoveryProven, selectionSafe bool, cost batchCost) (bool, []string) {
	var reasons []string
	if !recoveryProven {
		reasons = append(reasons, "unproven crash recovery")
	}
	if !selectionSafe {
		reasons = append(reasons, "unsafe change selection")
	}
	if cost.Reused == 0 || cost.ExecutedNS >= cost.StepNS {
		reasons = append(reasons, "no measured payback")
	}
	return len(reasons) == 0, reasons
}

// TestBatchPromotionControl proves batch promotion is admitted only with proven
// recovery, safe selection and measured payback, and that each missing
// precondition keeps the sequential floor with a named reason.
func TestBatchPromotionControl(t *testing.T) {
	// Reuse saved 600ns of the 1000ns summed step time: measured payback.
	payback := batchCost{Reused: 3, Accepted: 5, StepNS: 1000, ExecutedNS: 400}
	if promote, reasons := batchPromotionControl(true, true, payback); !promote || len(reasons) != 0 {
		t.Fatalf("full preconditions did not promote batching: %v", reasons)
	}
	for name, tc := range map[string]struct {
		recovery  bool
		selection bool
		cost      batchCost
		want      string
	}{
		"unproven recovery":    {false, true, payback, "unproven crash recovery"},
		"unsafe selection":     {true, false, payback, "unsafe change selection"},
		"no reuse":             {true, true, batchCost{Reused: 0, StepNS: 1000, ExecutedNS: 1000}, "no measured payback"},
		"reuse without saving": {true, true, batchCost{Reused: 2, StepNS: 500, ExecutedNS: 500}, "no measured payback"},
	} {
		t.Run(name, func(t *testing.T) {
			promote, reasons := batchPromotionControl(tc.recovery, tc.selection, tc.cost)
			if promote {
				t.Fatalf("%s promoted batching", name)
			}
			if !slices.Contains(reasons, tc.want) {
				t.Fatalf("%s: reasons %v missing %q", name, reasons, tc.want)
			}
		})
	}
}
