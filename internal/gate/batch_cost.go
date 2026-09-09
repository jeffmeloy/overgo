package gate

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// checkpointCost records one acceptance step's outcome and wall.
type checkpointCost struct {
	Name       string                `json:"name"`
	Outcome    runrecord.StepOutcome `json:"outcome"`
	DurationNS uint64                `json:"duration_ns"`
}

// batchCost separates summed step time from acceptance execution and reuse.
type batchCost struct {
	Checkpoints []checkpointCost `json:"checkpoints"`
	StepNS      uint64           `json:"step_ns"`
	Failed      int              `json:"failed"`
	FailedNS    uint64           `json:"failed_ns"`
	Other       int              `json:"other"`
	OtherNS     uint64           `json:"other_ns"`
	Accepted    int              `json:"accepted"`
	Reused      int              `json:"reused"`
	ExecutedNS  uint64           `json:"executed_ns"`
}

func acceptanceStep(name string) bool {
	return name == "acceptance" || strings.HasPrefix(name, "acceptance-")
}

// batchCostOf sums step durations; overlapping steps are not elapsed wall.
// Failures of any phase are counted apart; executed
// non-acceptance steps are the other phases; checkpoints sort by name.
func batchCostOf(steps []runrecord.GateStep) batchCost {
	var cost batchCost
	for _, step := range steps {
		cost.StepNS += step.DurationNS
		if step.Outcome == runrecord.StepFailed {
			cost.Failed++
			cost.FailedNS += step.DurationNS
		}
		if !acceptanceStep(step.Name) {
			if step.Outcome == runrecord.StepSucceeded {
				cost.Other++
				cost.OtherNS += step.DurationNS
			}
			continue
		}
		cost.Checkpoints = append(cost.Checkpoints, checkpointCost{Name: step.Name, Outcome: step.Outcome, DurationNS: step.DurationNS})
		switch step.Outcome {
		case runrecord.StepSucceeded:
			cost.Accepted++
			cost.ExecutedNS += step.DurationNS
		case runrecord.StepReused:
			cost.Accepted++
			cost.Reused++
		}
	}
	slices.SortFunc(cost.Checkpoints, func(left, right checkpointCost) int { return strings.Compare(left.Name, right.Name) })
	return cost
}

// reuseSavings estimates step time avoided, not elapsed wall saved: for each reused
// checkpoint, the lower median executed duration of that checkpoint across
// prior costs (the conservative estimate on an even count); a reused
// checkpoint never executed before is returned as unmeasured.
func reuseSavings(prior []batchCost, current batchCost) (uint64, []string) {
	executed := map[string][]uint64{}
	for _, cost := range prior {
		for _, checkpoint := range cost.Checkpoints {
			if checkpoint.Outcome == runrecord.StepSucceeded {
				executed[checkpoint.Name] = append(executed[checkpoint.Name], checkpoint.DurationNS)
			}
		}
	}
	var saved uint64
	var unmeasured []string
	for _, checkpoint := range current.Checkpoints {
		if checkpoint.Outcome != runrecord.StepReused {
			continue
		}
		durations := executed[checkpoint.Name]
		if len(durations) == 0 {
			unmeasured = append(unmeasured, checkpoint.Name)
			continue
		}
		slices.Sort(durations)
		saved += durations[(len(durations)-1)/2]
	}
	return saved, unmeasured
}

// priorBatchCosts reads the costs of every prior gate result recorded by an
// attempt of the same plan row; attempts whose result is not a gate result
// are skipped.
func priorBatchCosts(ctx context.Context, store *overgodb.Store, planRef string) ([]batchCost, error) {
	item, step, bound := strings.Cut(planRef, "/")
	if !bound {
		return nil, fmt.Errorf("gate: batch cost requires an item/step plan reference, got %q", planRef)
	}
	history, err := runrecord.LoadAttemptHistory(ctx, store, runrecord.AttemptFilter{PlanItem: item})
	if err != nil {
		return nil, err
	}
	var costs []batchCost
	for _, attempt := range history.Attempts {
		if attempt.PlanStep != step || !attempt.Result.Valid() {
			continue
		}
		result, err := runrecord.RequireGateResult(ctx, store, attempt.Result)
		if err != nil {
			continue
		}
		costs = append(costs, batchCostOf(result.Steps))
	}
	return costs, nil
}

// batchCostAudit reports measured wall, summed work and estimated avoided work
// on every gate; without acceptance steps the accepted figures read zero.
func (g *gateContext) batchCostAudit(ctx context.Context, store *overgodb.Store, steps []runrecord.GateStep, wallNS uint64) {
	current := batchCostOf(steps)
	prior, err := priorBatchCosts(ctx, store, g.planRef)
	if err != nil {
		g.audit = append(g.audit, "gate cost: prior attempts unavailable: "+err.Error())
		return
	}
	saved, unmeasured := reuseSavings(prior, current)
	g.audit = append(g.audit, fmt.Sprintf(
		"gate cost: total_wall=%s summed_step_time=%s failed=%d/%s other_phases=%d/%s accepted=%d reused=%d accepted_executed=%s estimated_step_time_avoided=%s prior_runs=%d unmeasured=%q",
		time.Duration(wallNS), time.Duration(current.StepNS), current.Failed, time.Duration(current.FailedNS), current.Other, time.Duration(current.OtherNS),
		current.Accepted, current.Reused, time.Duration(current.ExecutedNS), time.Duration(saved), len(prior), unmeasured,
	))
}
