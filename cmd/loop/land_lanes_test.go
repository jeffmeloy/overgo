package main

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// TestLandOutcomeNamesLaneObligation binds what a landing says about its
// lanes: a gate that deferred none owes none; one that deferred lanes is
// answered by the obligation naming the landed commit, which the typed
// outcome carries whatever its state; and a deferral with no obligation for
// the commit -- none at all, or only an earlier commit's -- fails the
// landing instead of reading as passed.
func TestLandOutcomeNamesLaneObligation(t *testing.T) {
	t.Parallel()
	const commit = "0123456789abcdef0123456789abcdef01234567"
	obligation := runrecord.GateLaneObligation{
		State: runrecord.LaneObligationPassed, CodeCommit: commit, Checks: []string{"test-device", "webui-lane"},
		ID: mustParseID(t, "evidence:sha256:"+strings.Repeat("a", 64)),
	}
	if report, failure := laneVerdict(nil, commit, runrecord.GateLaneObligation{}, false); report != nil || failure != "" {
		t.Fatalf("a gate that deferred nothing owed lanes: %+v %q", report, failure)
	}
	deferred := []string{"test-device", "webui-lane"}
	report, failure := laneVerdict(deferred, commit, obligation, true)
	if failure != "" || report == nil || report.Obligation != obligation.ID || report.State != string(runrecord.LaneObligationPassed) ||
		report.Commit != commit || !slices.Equal(report.Checks, obligation.Checks) {
		t.Fatalf("the owed obligation was not reported: %+v %q", report, failure)
	}
	earlier := obligation
	earlier.CodeCommit = strings.Repeat("f", 40)
	for name, candidate := range map[string]struct {
		obligation runrecord.GateLaneObligation
		found      bool
	}{"none": {}, "an earlier commit's": {earlier, true}} {
		if report, failure := laneVerdict(deferred, commit, candidate.obligation, candidate.found); report != nil || !strings.Contains(failure, "no lane obligation names commit") {
			t.Fatalf("%s obligation answered the deferral: %+v %q", name, report, failure)
		}
	}
	world := fakeRowWorld{report: report}
	outcome, err := landRow(&world, "item/do", "message")
	if err != nil || outcome.Outcome != outcomeReady || outcome.Lanes != report {
		t.Fatalf("the landing did not carry its lane obligation: %+v %v", outcome, err)
	}
	encoded, err := json.Marshal(outcome)
	if err != nil || !strings.Contains(string(encoded), `"lanes":{"obligation":"`+obligation.ID.String()+`","state":"passed"`) {
		t.Fatalf("the typed outcome does not name the obligation: %s %v", encoded, err)
	}
}

func mustParseID(t *testing.T, value string) artifact.ID {
	t.Helper()
	id, err := artifact.ParseID(value)
	if err != nil {
		t.Fatal(err)
	}
	return id
}
