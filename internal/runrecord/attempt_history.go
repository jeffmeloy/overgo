package runrecord

import (
	"context"
	"errors"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// Attempt history is the durable cross-run measurement surface: typed
// store reads over the attempt records every gate run emits, filtered
// and aggregated for steering. Nothing here scrapes logs; the store's
// media-typed query is the only source.

// AttemptFilter bounds one history read. Empty fields admit all.
type AttemptFilter struct {
	PlanItem   string
	Strategy   string
	CodeCommit string
	Limit      int
}

// AttemptAggregate is one plan step's measured record across attempts.
type AttemptAggregate struct {
	PlanItem    string `json:"plan_item"`
	PlanStep    string `json:"plan_step"`
	Attempts    int    `json:"attempts"`
	Succeeded   int    `json:"succeeded"`
	TotalWallNS uint64 `json:"total_wall_ns"`
	TotalFiles  int    `json:"total_files"`
}

// AttemptHistory carries the matched attempts and their per-step
// aggregates, both deterministically ordered.
type AttemptHistory struct {
	Attempts []AttemptRecord    `json:"attempts"`
	Steps    []AttemptAggregate `json:"steps"`
}

// attemptHistoryMaxRecords bounds one history read: steering consumes
// aggregates and recent attempts, never an unbounded scan.
const attemptHistoryMaxRecords = 4096

// LoadAttemptHistory reads every attempt record matching the filter
// from the store and derives the per-step aggregates.
func LoadAttemptHistory(ctx context.Context, store *overgodb.Store, filter AttemptFilter) (AttemptHistory, error) {
	if ctx == nil || store == nil {
		return AttemptHistory{}, errors.New("run record: attempt history requires the store")
	}
	limit := filter.Limit
	if limit <= 0 || limit > attemptHistoryMaxRecords {
		limit = attemptHistoryMaxRecords
	}
	result, err := store.Query(ctx, overgodb.Query{
		Kind: artifact.KindEvidence, MediaType: AttemptMediaType, Schema: AttemptSchema,
		MaxResults: attemptHistoryMaxRecords, Projection: overgodb.ProjectContentPresence,
	})
	if err != nil {
		return AttemptHistory{}, err
	}
	history := AttemptHistory{}
	for _, content := range result.Contents {
		record, found, err := attemptCodec.Read(ctx, store, content.Artifact)
		if err != nil {
			return AttemptHistory{}, err
		}
		if !found {
			continue
		}
		if filter.PlanItem != "" && record.PlanItem != filter.PlanItem {
			continue
		}
		if filter.Strategy != "" && record.Strategy != filter.Strategy {
			continue
		}
		if filter.CodeCommit != "" && record.CodeCommit != filter.CodeCommit {
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

// aggregateAttempts reduces attempts to one measured row per plan step.
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
