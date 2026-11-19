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

// TestAClosesLanesRunWhatItsChangesReach holds a run of the cycle's lanes on a row's close (validate -run -lanes
// -reached, lanes-reach) to running what the paths changed since each lane's last pass reach, read on the repository as
// a commit's lanes stage reads its own: a change to cmd/loop runs the models lane's cmd/loop package alone, a change to
// the plan runs no lane, a change to go.mod runs its lane whole, and a lane with no pass kept, or whose pass's commit
// cannot be read, runs whole. Every close had run all five lanes, five hours beside the day's landings.
func TestAClosesLanesRunWhatItsChangesReach(t *testing.T) {
	t.Parallel()
	runs := map[string]plan.LaneRecord{
		"models": {Passed: plan.LaneRun{Commit: "models-pass"}}, "audio": {Passed: plan.LaneRun{Commit: "audio-pass"}},
		"sequences": {Passed: plan.LaneRun{Commit: "sequences-pass"}}, "needle-reference": {Passed: plan.LaneRun{Commit: "pruned"}},
		"model-journey": {},
	}
	changed := changedSinceLastPass(runs, func(base string) ([]string, error) {
		switch base {
		case "models-pass":
			return []string{"cmd/loop/lanes_on_close.go"}, nil
		case "audio-pass":
			return []string{"docs/plan.json"}, nil
		case "sequences-pass":
			return []string{"go.mod"}, nil
		}
		return nil, errors.New("unknown revision")
	})
	if _, read := changed["needle-reference"]; read || len(changed) != 3 {
		t.Fatalf("the paths changed since each lane's pass read %v; want the three lanes whose pass is kept and readable", changed)
	}
	lanes := testcheck.Must(landing.CycleLanesReached(testutil.RepoRoot(t), changed))(t)
	all := landing.CycleLanes()
	byName := func(lanes []landing.CycleLane, name string) (landing.CycleLane, bool) {
		index := slices.IndexFunc(lanes, func(lane landing.CycleLane) bool { return lane.Name == name })
		if index < 0 {
			return landing.CycleLane{}, false
		}
		return lanes[index], true
	}
	if models, held := byName(lanes, "models"); !held || len(models.Each) != 1 || models.Each[0][len(models.Each[0])-1] != "./cmd/loop" {
		t.Errorf("a change to cmd/loop runs the models lane as %+v (held %t); want its cmd/loop package alone", models.Each, held)
	}
	if _, held := byName(lanes, "audio"); held {
		t.Error("a change to the plan alone runs the audio lane; want no lane")
	}
	for _, whole := range []string{"sequences", "needle-reference", "model-journey"} {
		lane, held := byName(lanes, whole)
		want, _ := byName(all, whole)
		if !held || !slices.Equal(lane.Arguments, want.Arguments) || len(lane.Each) != len(want.Each) {
			t.Errorf("the %s lane runs as %+v (held %t); want it whole, %+v", whole, lane, held, want)
		}
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

// TestAModelsLanePackageRunsWhatChangedSinceItsOwnPass holds a close's models lane to each package's own last pass
// (lanes-package-pass), read on the repository: where the lane last passed long ago and changes since then reach
// internal/latentimage and cmd/loop, but cmd/loop passed since, at a commit after which nothing changed, the lane runs
// internal/latentimage and not cmd/loop; and where a path cmd/loop takes in changed after its own pass, it runs both. A
// lane that failed in one package had kept no pass of the others, so every close ran them all against the lane's last
// pass (the models lane's at 8dfa48064 on 2026-11-16, every run since full-size).
func TestAModelsLanePackageRunsWhatChangedSinceItsOwnPass(t *testing.T) {
	t.Parallel()
	sinceLane := []string{"internal/latentimage/rotary_joint.go", "cmd/loop/main.go"}
	quiet := map[string]plan.LaneRecord{}
	for _, lane := range landing.CycleLanes() {
		quiet[lane.Name] = plan.LaneRecord{Passed: plan.LaneRun{Commit: "quiet", Passed: true}}
	}
	reading := func(sinceLoop []string) []string {
		runs := maps.Clone(quiet)
		runs["models"] = plan.LaneRecord{
			Passed:   plan.LaneRun{Commit: "lane-pass", Passed: true},
			Packages: map[string]plan.LaneRun{"./cmd/loop": {Commit: "loop-pass", Passed: true}},
		}
		lanes := testcheck.Must(reachedLanes(testutil.RepoRoot(t), runs, func(base string) ([]string, error) {
			switch base {
			case "quiet":
				return nil, nil
			case "lane-pass":
				return sinceLane, nil
			case "loop-pass":
				return sinceLoop, nil
			}
			return nil, errors.New("unknown revision")
		}))(t)
		var packages []string
		for _, lane := range lanes {
			if lane.Name != "models" {
				t.Fatalf("the lane %s runs though nothing changed since its pass", lane.Name)
			}
			for _, arguments := range lane.Each {
				packages = append(packages, arguments[len(arguments)-1])
			}
		}
		return packages
	}
	if packages := reading(nil); !slices.Contains(packages, "./internal/latentimage") || slices.Contains(packages, "./cmd/loop") {
		t.Errorf("with nothing changed since cmd/loop's own pass the models lane runs %v; want internal/latentimage and not cmd/loop", packages)
	}
	if packages := reading([]string{"cmd/loop/main.go"}); !slices.Contains(packages, "./internal/latentimage") || !slices.Contains(packages, "./cmd/loop") {
		t.Errorf("with cmd/loop changed since its own pass the models lane runs %v; want internal/latentimage and cmd/loop", packages)
	}
	// A run keeps the passes reading reads: of a lane's two packages, the one whose test passed beside a declared
	// exclusion passed, and the one whose test failed did not.
	passed, verdict := laneVerdict([]laneStep{{pkg: 0}, {pkg: 0}, {pkg: 1}}, []error{nil, errInapplicable, errors.New("broken")})
	if !slices.Equal(passed, []int{0}) || verdict == nil {
		t.Fatalf("a lane of a passing and a failing package read %v passed (%v); want the first alone", passed, verdict)
	}
	kept := t.TempDir()
	testcheck.NoError(t, plan.RecordLanePackages(kept, "models", plan.LaneRun{Commit: "loop-pass", Passed: true}, []string{"./cmd/loop"}))
	if own := plan.ReadLaneRuns(kept)["models"].Packages["./cmd/loop"]; own.Commit != "loop-pass" {
		t.Errorf("a package's pass was kept as %+v; want its commit", own)
	}
}
