package main

import (
	"errors"
	"maps"
	"slices"
	"testing"

	"overgo/internal/landing"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestALaneTestRunsAgainOnlyWhenWhatItReadsChanged holds a close's run of the lanes to each test's own last result
// (lanes-record), read on the repository: a test of the models lane that passed at a commit since which only the plan
// changed is left out, and one since which a file its package reads changed is taken; a failed test, a test with no
// result and a result at a commit whose changes cannot be read are taken; a lane run whole is left out on its own pass
// as a test is. Passes had been kept per lane and per package, a package's only once every one of its tests ran in one
// run, so TestMOSSGenerationCheckpoint ran in 7 runs from 19:12 on 2026-11-24 to 04:03 the next day, between three of
// which nothing it reads changed.
func TestALaneTestRunsAgainOnlyWhenWhatItReadsChanged(t *testing.T) {
	t.Parallel()
	results := map[string]plan.TestResult{
		testKey("./internal/latentimage", "TestQuiet"):      {Commit: "plan-only"},
		testKey("./internal/latentimage", "TestReached"):    {Commit: "moss-changed"},
		testKey("./internal/latentimage", "TestFailed"):     {Commit: "plan-only", Failed: true},
		testKey("./internal/latentimage", "TestUnreadable"): {Commit: "pruned"},
		testKey("", "sequences"):                            {Commit: "plan-only"},
		testKey("", "needle-reference"):                     {Commit: "moss-changed"},
	}
	current := stillCurrent(testutil.RepoRoot(t), results, func(base string) ([]string, error) {
		switch base {
		case "plan-only":
			return []string{"docs/plan.json"}, nil
		case "moss-changed":
			return []string{"internal/latentimage/rotary_joint.go"}, nil
		}
		return nil, errors.New("unknown revision")
	})
	for test, want := range map[string]bool{"TestQuiet": true, "TestReached": false, "TestFailed": false, "TestUnreadable": false, "TestMissing": false} {
		if got := current("models", "./internal/latentimage", testKey("./internal/latentimage", test)); got != want {
			t.Errorf("the models lane's %s is current: %t; want %t", test, got, want)
		}
	}
	if !current("sequences", "", testKey("", "sequences")) {
		t.Error("the sequences lane, passed at a commit since which only the plan changed, is run again; want it left out")
	}
	if current("model-journey", "", testKey("", "model-journey")) {
		t.Error("the browser lane, with no result kept, is left out; want it run")
	}
}

// TestARepositoryDataFileReachesTheLanesThatReadIt holds a close's lanes to the checks that read a changed data file
// beneath no Go package (lanes-fixtures), read on the repository: a change to fixtures/needle_upstream_calls.json runs
// the needle-reference lane, whose seq2seq tests read it, and neither the sequences nor the browser lane; a change to
// fixtures/moss/checkpoint.json runs the models lane's internal/latentimage, whose golden provenance test reads it, and
// not internal/mediacapability, which imports latentimage and reads no fixture; and a change to the plan and the
// findings register runs no lane. Both fixtures had reached no lane, data reaching a lane only beneath a package of its
// closure (SQA-415).
func TestARepositoryDataFileReachesTheLanesThatReadIt(t *testing.T) {
	t.Parallel()
	needle, moss := "fixtures/needle_upstream_calls.json", "fixtures/moss/checkpoint.json"
	lanes := testcheck.Must(landing.CycleLanesReached(testutil.RepoRoot(t), map[string][]string{
		"needle-reference": {needle}, "models": {moss}, "sequences": {needle}, "model-journey": {needle, moss},
		"audio": {"docs/plan.json", "docs/sqa_findings.json"},
	}))(t)
	held := map[string][]string{}
	for _, lane := range lanes {
		held[lane.Name] = nil
		for _, arguments := range lane.Each {
			held[lane.Name] = append(held[lane.Name], arguments[len(arguments)-1])
		}
	}
	if _, ran := held["needle-reference"]; !ran {
		t.Errorf("a change to %s runs the lanes %v; want the needle-reference lane", needle, slices.Sorted(maps.Keys(held)))
	}
	if models, ran := held["models"]; !ran || !slices.Contains(models, "./internal/latentimage") || slices.Contains(models, "./internal/mediacapability") {
		t.Errorf("a change to %s runs the models lane's %v (run %t); want internal/latentimage, which reads it, and not internal/mediacapability", moss, models, ran)
	}
	for _, unread := range []string{"sequences", "model-journey", "audio"} {
		if _, ran := held[unread]; ran {
			t.Errorf("the %s lane runs for files none of its packages reads; want it left out", unread)
		}
	}
}
