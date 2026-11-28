package main

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/scorecard"
	"overgo/internal/testcheck"
)

// TestARetryReachesItsCellsFromTheCommitBefore holds a run's reach to the record of an earlier commit (SQA-524): with a
// record of an earlier commit and, published after it, a partial one of the commit itself, as a cycle whose cells failed
// publishes, a run at the commit reaches its cells from the earlier one, as its first attempt did, so the cell that
// failed is selected again. Reached from its own partial record, the retry had found nothing changed, measured no cell
// and passed its cells phase. With no record of another commit, every cell is reached.
func TestARetryReachesItsCellsFromTheCommitBefore(t *testing.T) {
	t.Parallel()
	before, head := strings.Repeat("b", 40), strings.Repeat("a", 40)
	store := t.TempDir()
	at := time.Date(2026, time.November, 28, 3, 0, 0, 0, time.UTC)
	testcheck.Must(scorecard.Publish(scorecard.Directory(store), at, scorecard.Record{Commit: head}))(t)
	if from := testcheck.Must(reachedFrom(store, head))(t); from != nil {
		t.Errorf("with only its own commit's record, a run reaches its cells from %q; want no record, every cell reached", from.Commit)
	}
	testcheck.Must(scorecard.Publish(scorecard.Directory(store), at.Add(time.Hour), scorecard.Record{Commit: before}))(t)
	testcheck.Must(scorecard.Publish(scorecard.Directory(store), at.Add(2*time.Hour), scorecard.Record{Commit: head}))(t)
	if from := testcheck.Must(reachedFrom(store, head))(t); from == nil || from.Commit != before {
		t.Errorf("a retry beside its own partial record reaches its cells from %v; want the earlier commit's record %s", from, before)
	}
}
