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

// CheckpointCost records one acceptance step's outcome and wall.
type CheckpointCost struct {
	Name       string                `json:"name"`
	Outcome    runrecord.StepOutcome `json:"outcome"`
	DurationNS uint64                `json:"duration_ns"`
}

// BatchCost records the accepted checkpoint cost of one gate run: the parent
// acceptance and every checkpoint step; reused steps are counted, executed
// steps are summed.
type BatchCost struct {
	Checkpoints []CheckpointCost `json:"checkpoints"`
	Accepted    int              `json:"accepted"`
	Reused      int              `json:"reused"`
	ExecutedNS  uint64           `json:"executed_ns"`
}

func acceptanceStep(name string) bool {
	return name == "acceptance" || strings.HasPrefix(name, "acceptance-")
}

// BatchCostOf derives the cost from a gate result's steps; non-acceptance
// steps are ignored; checkpoints sort by name.
func BatchCostOf(steps []runrecord.GateStep) BatchCost {
	var cost BatchCost
	for _, step := range steps {
		if !acceptanceStep(step.Name) {
			continue
		}
		cost.Checkpoints = append(cost.Checkpoints, CheckpointCost{Name: step.Name, Outcome: step.Outcome, DurationNS: step.DurationNS})
		switch step.Outcome {
		case runrecord.StepSucceeded:
			cost.Accepted++
			cost.ExecutedNS += step.DurationNS
		case runrecord.StepReused:
			cost.Accepted++
			cost.Reused++
		}
	}
	slices.SortFunc(cost.Checkpoints, func(left, right CheckpointCost) int { return strings.Compare(left.Name, right.Name) })
	return cost
}

// ReuseSavings estimates the wall a run saved by reuse: for each reused
// checkpoint, the lower median executed duration of that checkpoint across
// prior costs (the conservative estimate on an even count); a reused
// checkpoint never executed before is returned as unmeasured.
func ReuseSavings(prior []BatchCost, current BatchCost) (uint64, []string) {
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
func priorBatchCosts(ctx context.Context, store *overgodb.Store, planRef string) ([]BatchCost, error) {
	item, step, bound := strings.Cut(planRef, "/")
	if !bound {
		return nil, fmt.Errorf("gate: batch cost requires an item/step plan reference, got %q", planRef)
	}
	history, err := runrecord.LoadAttemptHistory(ctx, store, runrecord.AttemptFilter{PlanItem: item})
	if err != nil {
		return nil, err
	}
	var costs []BatchCost
	for _, attempt := range history.Attempts {
		if attempt.PlanStep != step || !attempt.Result.Valid() {
			continue
		}
		result, err := runrecord.RequireGateResult(ctx, store, attempt.Result)
		if err != nil {
			continue
		}
		costs = append(costs, BatchCostOf(result.Steps))
	}
	return costs, nil
}

// batchCostAudit appends this run's accepted checkpoint cost and the reuse
// saving against prior runs of the same row; no acceptance steps -> no line.
func (g *gateContext) batchCostAudit(ctx context.Context, store *overgodb.Store, steps []runrecord.GateStep) {
	current := BatchCostOf(steps)
	if current.Accepted == 0 {
		return
	}
	prior, err := priorBatchCosts(ctx, store, g.planRef)
	if err != nil {
		g.audit = append(g.audit, "gate cost: prior attempts unavailable: "+err.Error())
		return
	}
	saved, unmeasured := ReuseSavings(prior, current)
	g.audit = append(g.audit, fmt.Sprintf(
		"gate cost: accepted=%d reused=%d executed=%s saved=%s prior_runs=%d unmeasured=%q",
		current.Accepted, current.Reused, time.Duration(current.ExecutedNS), time.Duration(saved), len(prior), unmeasured,
	))
}
