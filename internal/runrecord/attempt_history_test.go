package runrecord

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestAttemptHistorySummary(t *testing.T) {
	strategy := testutil.ArtifactID(t, artifact.KindProfile, "strategy")
	trajectory := testutil.ArtifactID(t, artifact.KindEvidence, "trajectory")
	first := fixtureAttempt(t)
	first.StrategyID, first.Trajectory, first.CostUnits = strategy, trajectory, 5
	recoveryRoot, err := NewCausalRoot(TriggerStageWakeup, testutil.ArtifactID(t, artifact.KindEvidence, "history-root"))
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := recoveryRoot.Derive(TriggerRecovery, testutil.ArtifactID(t, artifact.KindEvidence, "lost-attempt"))
	if err != nil {
		t.Fatal(err)
	}
	first.Causal = &recovered
	first, err = NewAttemptRecord(first)
	if err != nil {
		t.Fatal(err)
	}
	second := fixtureAttempt(t)
	second.StrategyID, second.Outcome, second.Failure, second.CostUnits = strategy, OutcomeFailed, "tests", 3
	second, err = NewAttemptRecord(second)
	if err != nil {
		t.Fatal(err)
	}
	history := SummarizeAttemptHistory([]AttemptRecord{second, first}, AttemptHistoryFilter{StrategyID: strategy})
	if len(history) != 1 || history[0].Attempts != 2 || history[0].Succeeded != 1 || history[0].CostUnits != 8 || history[0].Recoveries != 1 || history[0].Trajectories[0] != trajectory {
		t.Fatalf("history = %+v", history)
	}
}

func commitAttempt(t *testing.T, store *overgodb.Store, item, step string, outcome Outcome, failure string, wall uint64) AttemptRecord {
	t.Helper()
	record := fixtureAttempt(t)
	record.PlanItem, record.PlanStep, record.Outcome, record.Failure, record.WallNS = item, step, outcome, failure, wall
	published, err := NewAttemptRecord(record)
	if err != nil {
		t.Fatal(err)
	}
	content, err := published.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.Commit(context.Background(), artifact.Batch{Key: "fixture/attempt/" + published.ID.String(), Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	return published
}

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
	if err != nil || len(history.Attempts) != 3 || len(history.Steps) != 2 {
		t.Fatalf("history = %+v, %v", history, err)
	}
	alpha := history.Steps[0]
	if alpha.PlanItem != "alpha" || alpha.Attempts != 2 || alpha.Succeeded != 1 || alpha.TotalWallNS != 140 {
		t.Fatalf("alpha = %+v", alpha)
	}
	filtered, err := LoadAttemptHistory(ctx, store, AttemptFilter{PlanItem: "beta"})
	if err != nil || len(filtered.Attempts) != 1 {
		t.Fatalf("item filter = %+v, %v", filtered, err)
	}
	byCommit, err := LoadAttemptHistory(ctx, store, AttemptFilter{CodeCommit: strings.Repeat("ab", 20)})
	if err != nil || len(byCommit.Attempts) != 3 {
		t.Fatalf("commit filter = %+v, %v", byCommit, err)
	}
	limited, err := LoadAttemptHistory(ctx, store, AttemptFilter{Limit: 1})
	if err != nil || len(limited.Attempts) != 1 || len(limited.Steps) != 1 {
		t.Fatalf("limit = %+v, %v", limited, err)
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
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/attempt/" + published.ID.String(), Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	byStrategy, err := LoadAttemptHistory(ctx, store, AttemptFilter{Strategy: "sonnet-baseline"})
	if err != nil || len(byStrategy.Attempts) != 1 || byStrategy.Attempts[0].PlanItem != "gamma" {
		t.Fatalf("strategy filter = %+v, %v", byStrategy, err)
	}
}
