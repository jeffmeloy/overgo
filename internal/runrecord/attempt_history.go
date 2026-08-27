package runrecord

import (
	"slices"
	"sort"

	"overgo/internal/artifact"
)

type AttemptHistoryFilter struct {
	PlanItem     string
	Strategy     artifact.ID
	CodeCommit   string
	TaskContract artifact.ID
	Environment  artifact.ID
}

type AttemptStepHistory struct {
	PlanItem             string        `json:"plan_item"`
	PlanStep             string        `json:"plan_step"`
	Attempts             uint64        `json:"attempts"`
	Succeeded            uint64        `json:"succeeded"`
	VerificationSelected uint64        `json:"verification_selected"`
	VerificationDefined  uint64        `json:"verification_defined"`
	EffectUncertainty    uint64        `json:"effect_uncertainty"`
	CostUnits            uint64        `json:"cost_units"`
	WallNS               uint64        `json:"wall_ns"`
	Churn                uint64        `json:"churn"`
	Recoveries           uint64        `json:"recoveries"`
	Trajectories         []artifact.ID `json:"trajectories,omitempty"`
}

// AttemptHistory aggregates parsed attempt artifacts; repository indexes still
// own discovery, and no log or parallel authority is consulted.
func AttemptHistory(records []AttemptRecord, filter AttemptHistoryFilter) []AttemptStepHistory {
	byStep := map[string]*AttemptStepHistory{}
	for _, record := range records {
		if filter.PlanItem != "" && record.PlanItem != filter.PlanItem || filter.Strategy.Valid() && record.Strategy != filter.Strategy ||
			filter.CodeCommit != "" && record.CodeCommit != filter.CodeCommit || filter.TaskContract.Valid() && record.TaskContract != filter.TaskContract ||
			filter.Environment.Valid() && record.Environment != filter.Environment {
			continue
		}
		key := record.PlanItem + "\x00" + record.PlanStep
		summary := byStep[key]
		if summary == nil {
			summary = &AttemptStepHistory{PlanItem: record.PlanItem, PlanStep: record.PlanStep}
			byStep[key] = summary
		}
		summary.Attempts++
		if record.Outcome == OutcomeSucceeded {
			summary.Succeeded++
		}
		summary.VerificationSelected += uint64(record.Selection.Selected)
		summary.VerificationDefined += uint64(record.Selection.Defined)
		summary.EffectUncertainty += uint64(record.Selection.Uncertainty)
		summary.CostUnits += record.CostUnits
		summary.WallNS += record.WallNS
		summary.Churn += uint64(record.Diff.Insertions) + uint64(record.Diff.Deletions)
		if record.Recovered {
			summary.Recoveries++
		}
		if record.Trajectory.Valid() {
			summary.Trajectories = append(summary.Trajectories, record.Trajectory)
		}
	}
	result := make([]AttemptStepHistory, 0, len(byStep))
	for _, summary := range byStep {
		sort.Slice(summary.Trajectories, func(i, j int) bool { return artifact.CompareID(summary.Trajectories[i], summary.Trajectories[j]) < 0 })
		summary.Trajectories = slices.Compact(summary.Trajectories)
		result = append(result, *summary)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].PlanItem != result[j].PlanItem {
			return result[i].PlanItem < result[j].PlanItem
		}
		return result[i].PlanStep < result[j].PlanStep
	})
	return result
}
