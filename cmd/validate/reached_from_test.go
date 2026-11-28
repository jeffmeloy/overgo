package main

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/scorecard"
	"overgo/internal/testcheck"
)

// TestARunIsComparedWithAnEarlierCommitsRecord holds a run to the record of an earlier commit (SQA-524), the record its
// contracts join and against which it reaches and chooses its cells: with an earlier commit's record and, published
// after it, a partial one of the checked-out commit, as a cycle whose cells failed publishes, a run at that commit is
// compared with the earlier one, as its first attempt was, so the cell that failed is reached and chosen again.
// Compared with its own partial record, a retry had found nothing changed, measured no cell, and dropped a failed first
// reading whose model and task another cell's reading there answered, while its cells phase passed. With only its own
// commit's record, every cell is reached; outside a checkout, the newest record is the one.
func TestARunIsComparedWithAnEarlierCommitsRecord(t *testing.T) {
	t.Parallel()
	root, head := fixtureRepository(t)
	store, at := t.TempDir(), time.Date(2026, time.November, 28, 3, 0, 0, 0, time.UTC)
	testcheck.Must(scorecard.Publish(scorecard.Directory(store), at, scorecard.Record{Commit: head}))(t)
	if from := testcheck.Must(reachedFrom(t.Context(), root, store))(t); from != nil {
		t.Errorf("with only its own commit's record, a run is compared with %q; want no record, every cell reached", from.Commit)
	}
	before := strings.Repeat("b", 40)
	testcheck.Must(scorecard.Publish(scorecard.Directory(store), at.Add(time.Hour), scorecard.Record{Commit: before}))(t)
	testcheck.Must(scorecard.Publish(scorecard.Directory(store), at.Add(2*time.Hour), scorecard.Record{Commit: head}))(t)
	if from := testcheck.Must(reachedFrom(t.Context(), root, store))(t); from == nil || from.Commit != before {
		t.Errorf("a retry beside its own partial record is compared with %v; want the earlier commit's record %s", from, before)
	}
	if from := testcheck.Must(reachedFrom(t.Context(), t.TempDir(), store))(t); from == nil || from.Commit != head {
		t.Errorf("outside a checkout a run is compared with %v; want the newest record, of %s", from, head)
	}
}
