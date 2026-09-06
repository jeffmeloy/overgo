package loop

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// replayReceipt commits one succeeded gate attempt for the row at commit
// and returns the attempt and its gate result.
func replayReceipt(t *testing.T, store *overgodb.Store, row Step, commit string) (runrecord.AttemptRecord, runrecord.GateResult) {
	t.Helper()
	steps := []runrecord.GateStep{
		{Name: "protection", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepSucceeded, DurationNS: 3},
		{Name: "test", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 7},
	}
	record, err := runrecord.NewGateRecord(
		testutil.ArtifactID(t, artifact.KindRecipe, "replay recipe"),
		testutil.ArtifactID(t, artifact.KindEvidence, "replay environment"),
		commit, runrecord.OutcomeSucceeded, "", 10, steps,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch("replay/gate/" + record.Result.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: row.Item, PlanStep: row.ID, Result: record.Result.ID, Recipe: record.Result.Recipe,
		CodeCommit: commit, Outcome: runrecord.OutcomeSucceeded, WallNS: 10,
		Selection: runrecord.AttemptSelection{Defined: 2, Selected: 2}, Diff: runrecord.AttemptDiff{Files: 1, Insertions: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := attempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: record.Result.Recipe}, artifact.Descriptor{ID: record.Result.Environment}, content.Descriptor)
	batch.Contents = append(batch.Contents, content)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return attempt, record.Result
}

// TestReplayAttemptFromReceipt pins: a replay with unchanged inputs reads
// every recorded step from the receipt without executing; a replay with
// changed inputs executes every step and logs the new results under its
// invocation; a third replay with the changed inputs reads those logged
// results; a receipt of another row is refused; invocations count up.
func TestReplayAttemptFromReceipt(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	row := Step{Item: "guard", ID: "check"}
	commit := strings.Repeat("ab", 20)
	receipt, result := replayReceipt(t, store, row, commit)
	executed := 0
	// A re-executed step records the receipt result itself: the log only
	// needs a committed artifact per key.
	execute := func(context.Context, runrecord.GateStep) (artifact.ID, error) {
		executed++
		return result.ID, nil
	}
	same, err := ReplayAttempt(ctx, store, row, receipt, result, commit, execute)
	if err != nil || same.Invocation != 1 || len(same.Reused) != 2 || len(same.Executed) != 0 || executed != 0 {
		t.Fatalf("unchanged inputs replay = %+v executed=%d, %v", same, executed, err)
	}
	changed := strings.Repeat("cd", 20)
	differing, err := ReplayAttempt(ctx, store, row, receipt, result, changed, execute)
	if err != nil || differing.Invocation != 2 || len(differing.Executed) != 2 || len(differing.Reused) != 0 || executed != 2 {
		t.Fatalf("changed inputs replay = %+v executed=%d, %v", differing, executed, err)
	}
	again, err := ReplayAttempt(ctx, store, row, receipt, result, changed, execute)
	if err != nil || again.Invocation != 3 || len(again.Reused) != 2 || executed != 2 {
		t.Fatalf("repeated changed inputs replay = %+v executed=%d, %v", again, executed, err)
	}
	if _, err := ReplayAttempt(ctx, store, Step{Item: "other", ID: "row"}, receipt, result, commit, execute); err == nil {
		t.Fatal("a receipt of another row was replayed")
	}
	if _, err := ReplayAttempt(ctx, store, row, receipt, result, " ", execute); err == nil {
		t.Fatal("a replay without an inputs identity was accepted")
	}
}
