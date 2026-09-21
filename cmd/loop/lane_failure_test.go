package main

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestLaneFailureSaysWhatFailed holds the driver's report of a failed lane
// obligation to starting the repair from the failure: beside the commit and
// the result to inspect it names the lane that failed and what that lane
// said, read from the recorded result; when the result cannot be read it
// still says which result to inspect.
func TestLaneFailureSaysWhatFailed(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const said = "running tests: TestWebUIBrowserVideoCapture (19m58s)"
	record, err := runrecord.NewGateRecord(
		testutil.ArtifactID(t, artifact.KindRecipe, "lanes"), testutil.ArtifactID(t, artifact.KindEvidence, "environment"),
		strings.Repeat("a", 40), runrecord.OutcomeFailed, "webui-lane", 1,
		[]runrecord.GateStep{
			{Name: "test-device", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1},
			{Name: "webui-lane", Phase: runrecord.PhaseTest, Outcome: runrecord.StepFailed, DurationNS: 1, Detail: said},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	content, err := record.Result.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "lane-failure/result", Contents: []artifact.Content{content}}); err != nil {
		t.Fatal(err)
	}
	obligation := runrecord.GateLaneObligation{CodeCommit: strings.Repeat("a", 40), Outcome: record.Result.ID}
	if debt := laneFailure(store, obligation); !strings.Contains(debt, record.Result.ID.String()) || !strings.Contains(debt, "webui-lane said: "+said) {
		t.Fatalf("debt = %q, want the result and what the failed lane said", debt)
	}
	obligation.Outcome = testutil.ArtifactID(t, artifact.KindEvidence, "absent result")
	if debt := laneFailure(store, obligation); !strings.Contains(debt, obligation.Outcome.String()) || strings.Contains(debt, " said: ") {
		t.Fatalf("debt for an unreadable result = %q, want the result named and nothing invented", debt)
	}
}
