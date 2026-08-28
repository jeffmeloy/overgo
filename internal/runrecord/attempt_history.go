package runrecord

import (
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// AttemptFilter narrows loaded attempt records; zero fields match everything.
type AttemptFilter struct {
	PlanItem   string
	Strategy   string
	StrategyID artifact.ID
	CodeCommit string
	Limit      int
}

// AttemptAggregate totals the attempts of one plan item and step.
type AttemptAggregate struct {
	PlanItem    string `json:"plan_item"`
	PlanStep    string `json:"plan_step"`
	Attempts    int    `json:"attempts"`
	Succeeded   int    `json:"succeeded"`
	TotalWallNS uint64 `json:"total_wall_ns"`
	TotalFiles  int    `json:"total_files"`
}

// AttemptHistory pairs filtered attempt records with their per-step aggregates.
type AttemptHistory struct {
	Attempts []AttemptRecord    `json:"attempts"`
	Steps    []AttemptAggregate `json:"steps"`
}

// LoadAttemptHistory queries committed attempt records from the store,
// filters and orders them deterministically, and aggregates them per step.
func LoadAttemptHistory(ctx context.Context, store *overgodb.Store, filter AttemptFilter) (AttemptHistory, error) {
	if ctx == nil || store == nil {
		return AttemptHistory{}, errors.New("run record: attempt history requires the store")
	}
	limit := filter.Limit
	if limit <= 0 || limit > MaximumAttemptPopulation {
		limit = MaximumAttemptPopulation
	}
	result, err := store.Query(ctx, overgodb.Query{Kind: artifact.KindEvidence, MediaType: AttemptMediaType, Schema: AttemptSchema, MaxResults: MaximumAttemptPopulation, Projection: overgodb.ProjectContentPresence})
	if err != nil {
		return AttemptHistory{}, err
	}
	history := AttemptHistory{}
	for _, content := range result.Contents {
		record, found, err := attemptCodec.Read(ctx, store, content.Artifact)
		if err != nil {
			return AttemptHistory{}, err
		}
		if !found || filter.PlanItem != "" && record.PlanItem != filter.PlanItem || filter.Strategy != "" && record.Strategy != filter.Strategy ||
			filter.StrategyID.Valid() && record.StrategyID != filter.StrategyID || filter.CodeCommit != "" && record.CodeCommit != filter.CodeCommit {
			continue
		}
		history.Attempts = append(history.Attempts, record)
	}
	sort.Slice(history.Attempts, func(i, j int) bool {
		left, right := history.Attempts[i], history.Attempts[j]
		if left.PlanItem != right.PlanItem {
			return left.PlanItem < right.PlanItem
		}
		if left.PlanStep != right.PlanStep {
			return left.PlanStep < right.PlanStep
		}
		return left.ID.String() < right.ID.String()
	})
	if len(history.Attempts) > limit {
		history.Attempts = history.Attempts[:limit]
	}
	history.Steps = aggregateAttempts(history.Attempts)
	return history, nil
}

func aggregateAttempts(attempts []AttemptRecord) []AttemptAggregate {
	type key struct{ item, step string }
	byStep := map[key]*AttemptAggregate{}
	order := make([]key, 0, len(attempts))
	for _, attempt := range attempts {
		k := key{attempt.PlanItem, attempt.PlanStep}
		aggregate, seen := byStep[k]
		if !seen {
			aggregate = &AttemptAggregate{PlanItem: k.item, PlanStep: k.step}
			byStep[k] = aggregate
			order = append(order, k)
		}
		aggregate.Attempts++
		if attempt.Outcome == OutcomeSucceeded {
			aggregate.Succeeded++
		}
		aggregate.TotalWallNS += attempt.WallNS
		aggregate.TotalFiles += attempt.Diff.Files
	}
	result := make([]AttemptAggregate, 0, len(order))
	for _, k := range order {
		result = append(result, *byStep[k])
	}
	return result
}

// AttemptHistoryFilter narrows attempt summaries by exact authorities; zero
// fields match everything.
type AttemptHistoryFilter struct {
	PlanItem     string
	StrategyID   artifact.ID
	CodeCommit   string
	TaskContract artifact.ID
	Environment  artifact.ID
}

// AttemptStepHistory totals one plan step's attempts, verification selection,
// cost, churn, recoveries, and trajectories.
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

// SummarizeAttemptHistory folds attempt records into ordered per-step summaries.
func SummarizeAttemptHistory(records []AttemptRecord, filter AttemptHistoryFilter) []AttemptStepHistory {
	byStep := map[string]*AttemptStepHistory{}
	for _, record := range records {
		if filter.PlanItem != "" && record.PlanItem != filter.PlanItem || filter.StrategyID.Valid() && record.StrategyID != filter.StrategyID ||
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
		if record.Causal != nil && record.Causal.Trigger == TriggerRecovery {
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
