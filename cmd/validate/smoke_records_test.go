package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestSmokeCurrentFollowsTheInferenceSurface holds a DNA model's currency to
// its smoke run and the inference surface: a passing run at the current
// commit is current, one at a commit the surface has moved past is not, a
// failed run is not, and no run says how to acquire one.
func TestSmokeCurrentFollowsTheInferenceSurface(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	model := testutil.ArtifactID(t, artifact.KindModel, "carbon")
	head, err := revParse(t.Context(), root, "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if err := smokeCurrent(t.Context(), root, model, runrecord.Run{Outcome: runrecord.OutcomeSucceeded, CodeCommit: head}, true); err != nil {
		t.Errorf("a passing run at the current commit is not current: %v", err)
	}
	// The staged Q1_0 prefill landed at 161b5138 and moved the executor; a run
	// at its parent stands for a surface the code has left.
	err = smokeCurrent(t.Context(), root, model, runrecord.Run{Outcome: runrecord.OutcomeSucceeded, CodeCommit: "36c838022b4993050e9386a7b6697d31280a27eb"}, true)
	if err == nil || !strings.Contains(err.Error(), "inference surface moved") {
		t.Errorf("a run before a surface move was judged current: %v", err)
	}
	err = smokeCurrent(t.Context(), root, model, runrecord.Run{Outcome: runrecord.OutcomeFailed, Failure: "smoke_failed", CodeCommit: head}, true)
	if err == nil || !strings.Contains(err.Error(), "failed") {
		t.Errorf("a failed run was judged current: %v", err)
	}
	err = smokeCurrent(t.Context(), root, model, runrecord.Run{}, false)
	if err == nil || !strings.Contains(err.Error(), "smoke-lane") {
		t.Errorf("no run did not name the smoke lane: %v", err)
	}
}
