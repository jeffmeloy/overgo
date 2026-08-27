package runrecord

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func commitAttempt(t *testing.T, store *overgodb.Store, item, step string, outcome Outcome, failure string, wall uint64) AttemptRecord {
	t.Helper()
	record := fixtureAttempt(t)
	record.PlanItem, record.PlanStep = item, step
	record.Outcome, record.Failure, record.WallNS = outcome, failure, wall
	published, err := NewAttemptRecord(record)
	if err != nil {
		t.Fatal(err)
	}
	content, err := published.Content()
	if err != nil {
		t.Fatal(err)
	}
	_, err = store.Commit(context.Background(), artifact.Batch{
		Key:       "fixture/attempt/" + published.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	})
	if err != nil {
		t.Fatal(err)
	}
	return published
}

// TestAttemptHistoryQueriesAndAggregates pins the steering surface:
// history reads only attempt-typed store documents, filters by plan
// item and code revision, and aggregates attempts, successes, and
// measured cost per step.
func TestAttemptHistoryQueriesAndAggregates(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	commitAttempt(t, store, "alpha", "do", OutcomeFailed, "magics", 100)
	commitAttempt(t, store, "alpha", "do", OutcomeSucceeded, "", 40)
	commitAttempt(t, store, "beta", "do", OutcomeSucceeded, "", 7)

	history, err := LoadAttemptHistory(ctx, store, AttemptFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(history.Attempts) != 3 || len(history.Steps) != 2 {
		t.Fatalf("history = %d attempts, %d steps", len(history.Attempts), len(history.Steps))
	}
	alpha := history.Steps[0]
	if alpha.PlanItem != "alpha" || alpha.Attempts != 2 || alpha.Succeeded != 1 || alpha.TotalWallNS != 140 {
		t.Fatalf("alpha aggregate = %+v", alpha)
	}

	filtered, err := LoadAttemptHistory(ctx, store, AttemptFilter{PlanItem: "beta"})
	if err != nil || len(filtered.Attempts) != 1 || filtered.Attempts[0].PlanItem != "beta" {
		t.Fatalf("item filter = (%+v, %v)", filtered.Attempts, err)
	}

	byCommit, err := LoadAttemptHistory(ctx, store, AttemptFilter{CodeCommit: strings.Repeat("ab", 20)})
	if err != nil || len(byCommit.Attempts) != 3 {
		t.Fatalf("commit filter = (%d, %v)", len(byCommit.Attempts), err)
	}
	if none, err := LoadAttemptHistory(ctx, store, AttemptFilter{CodeCommit: strings.Repeat("cd", 20)}); err != nil || len(none.Attempts) != 0 {
		t.Fatalf("mismatched commit filter = (%d, %v)", len(none.Attempts), err)
	}

	limited, err := LoadAttemptHistory(ctx, store, AttemptFilter{Limit: 1})
	if err != nil || len(limited.Attempts) != 1 || len(limited.Steps) != 1 {
		t.Fatalf("limit = (%d attempts, %d steps, %v)", len(limited.Attempts), len(limited.Steps), err)
	}

	strategist := fixtureAttempt(t)
	strategist.PlanItem, strategist.Strategy = "gamma", "sonnet-baseline"
	published, err := NewAttemptRecord(strategist)
	if err != nil {
		t.Fatal(err)
	}
	content, err := published.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "fixture/attempt/" + published.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	byStrategy, err := LoadAttemptHistory(ctx, store, AttemptFilter{Strategy: "sonnet-baseline"})
	if err != nil || len(byStrategy.Attempts) != 1 || byStrategy.Attempts[0].PlanItem != "gamma" {
		t.Fatalf("strategy filter = (%+v, %v)", byStrategy.Attempts, err)
	}
}
