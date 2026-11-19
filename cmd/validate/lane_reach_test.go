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
