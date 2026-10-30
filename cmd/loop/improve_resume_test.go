package main

import (
	"errors"
	"io"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"overgo/internal/clioptions"
)

// TestCycleReportsWhatItCreditedAndEachPhaseWall holds a cycle to saying what it did with
// the interrupted run it continued: a first run names each phase it ran, how it ended and its
// wall; the run that continues it names the phases it credited, and the checks it ran again,
// so no phase is skipped or redone in silence.
func TestCycleReportsWhatItCreditedAndEachPhaseWall(t *testing.T) {
	t.Parallel()
	state := filepath.Join(t.TempDir(), cycleStateFile)
	cycle := func(failing string) string {
		var report strings.Builder
		_ = measurePhases(state, "commit|store|obligations", &report, nil, nil, func(_ []string, arguments ...string) error {
			if strings.Join(arguments, " ") == failing {
				return errors.New("ended")
			}
			return nil
		})
		return report.String()
	}
	wall := regexp.MustCompile(`in [0-9.]+(µs|ms|s|m)`)
	first := cycle("-scorecard -publish")
	for _, want := range []string{"phase preflight passed", "phase lanes passed", "phase cells passed", "phase publication did not pass"} {
		if !strings.Contains(first, want) {
			t.Errorf("the first run's report %q does not say %q", first, want)
		}
	}
	if got := len(wall.FindAllString(first, -1)); got != 4 {
		t.Errorf("the first run's report %q gives %d phase walls, want one for each of four phases", first, got)
	}
	resumed := cycle("")
	for _, want := range []string{"phase preflight passed", "phase lanes passed", "phase cells credited from the interrupted run", "phase publication passed"} {
		if !strings.Contains(resumed, want) {
			t.Errorf("the resumed run's report %q does not say %q", resumed, want)
		}
	}
	if strings.Contains(resumed, "phase cells passed") || len(wall.FindAllString(resumed, -1)) != 3 {
		t.Errorf("the resumed run's report %q ran a credited phase again or gave other than the checks' and the publication's walls", resumed)
	}
}

// TestChecksRunBeforeTheCells holds a cycle to running the checks its cells depend on, each
// cell's preflight and then the lanes, before it measures anything. A lane that fails ends the
// cycle there, with no cell measured, no scorecard published and nothing filed. A preflight that
// refuses a cell names it first and the cycle goes on, since one model's missing precondition
// leaves the others' numbers sound. The cycle's error names the check that failed.
func TestChecksRunBeforeTheCells(t *testing.T) {
	t.Parallel()
	const preflight, lanes, cells, publish = "-run -preflight", "-run -lanes", "-run -cells", "-scorecard -publish"
	for failing, want := range map[string][]string{
		preflight: {preflight, lanes, cells, publish, "filed"},
		lanes:     {preflight, lanes},
		"":        {preflight, lanes, cells, publish, "filed"},
	} {
		var ran []string
		err := measurePhases(filepath.Join(t.TempDir(), cycleStateFile), "commit|store|obligations", io.Discard, func() error {
			ran = append(ran, "filed")
			return nil
		}, nil, func(_ []string, arguments ...string) error {
			phase := strings.Join(arguments, " ")
			ran = append(ran, phase)
			if phase == failing {
				return errors.New("refused")
			}
			return nil
		})
		if !slices.Equal(ran, want) || (err != nil) != (failing != "") || err != nil && !strings.Contains(err.Error(), failing) {
			t.Errorf("a failing %q: ran %q with error %v, want %q and only a failed check named", failing, ran, err, want)
		}
	}
}

// TestInterruptedCycleResumesAtItsFirstUnpassedPhase holds a cycle to going on
// from where it stopped for the same key: the cells that passed are not run
// again when a run ended in the publication, the checks always run (go test's
// cache replays the lanes that passed), a cell failure still publishes, and
// another key, another commit, store or obligation, starts over.
func TestInterruptedCycleResumesAtItsFirstUnpassedPhase(t *testing.T) {
	t.Parallel()
	state := filepath.Join(t.TempDir(), cycleStateFile)
	var ran []string
	var failing string
	validate := func(_ []string, arguments ...string) error {
		phase := strings.Join(arguments, " ")
		ran = append(ran, phase)
		if phase == failing {
			return errors.New("ended")
		}
		return nil
	}
	cycle := func(key, failPhase string) ([]string, error) {
		ran, failing = nil, failPhase
		err := measurePhases(state, key, io.Discard, nil, nil, validate)
		return ran, err
	}
	const preflight, lanes, cells, publish = "-run -preflight", "-run -lanes", "-run -cells", "-scorecard -publish"
	for _, step := range []struct {
		name, key, fail string
		want            []string
		wantErr         bool
	}{
		{"a cycle that ends in its publication", "commit-a|store|obligations", publish, []string{preflight, lanes, cells, publish}, true},
		{"the next run goes on at the publication", "commit-a|store|obligations", "", []string{preflight, lanes, publish}, false},
		{"a passed cycle runs only its checks", "commit-a|store|obligations", "", []string{preflight, lanes}, false},
		{"another store starts over", "commit-a|other-store|obligations", cells, []string{preflight, lanes, cells, publish}, true},
		{"so does an obligation that changed", "commit-a|other-store|activated", "", []string{preflight, lanes, cells, publish}, false},
		{"another commit starts over", "commit-b|other-store|activated", "", []string{preflight, lanes, cells, publish}, false},
	} {
		got, err := cycle(step.key, step.fail)
		if !slices.Equal(got, step.want) || (err != nil) != step.wantErr {
			t.Errorf("%s: ran %q with error %v, want %q (error %t)", step.name, got, err, step.want, step.wantErr)
		}
	}
}

// TestCycleCreditFollowsTheInputsARunConsumes holds the credit of a cycle to what a run reads: a
// change to the root an inventory was registered from, the committed data a training or
// projection run reads (grounding, objective, conditioning, projector), or what validate found
// the files a cell's command is given to be (consumed) changes the obligations the cycle is
// keyed by, as a changed recipe does, so the phases that passed run again; every phase a cycle
// runs is told the file its cells' credit is kept in, beside the cycle's state.
func TestCycleCreditFollowsTheInputsARunConsumes(t *testing.T) {
	t.Parallel()
	digest := func(cell string) string {
		t.Helper()
		value, err := obligationsDigest([]byte(`{"run":[{"model":"m","kind":"training","validation":"training-proof","recipe":"r1"` + cell + `}],"background":[]}`))
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	base := digest("")
	for field, value := range map[string]string{
		"directory": "d:/models/a", "grounding": "dataset:1", "objective": "objective.registered.text-text", "conditioning": "d:/video-stack", "projector": "d:/mmproj.gguf",
		"consumed": "what the files a command is given were",
	} {
		changed := digest(`,"` + field + `":"` + value + `"`)
		if changed == base {
			t.Errorf("a cell that names a %s left the obligations as they were", field)
		}
		if digest(`,"`+field+`":"`+value+`x"`) == changed {
			t.Errorf("a changed %s left the obligations as they were", field)
		}
	}

	// A cycle whose cells and publication passed runs them again once a file a cell's command is
	// given has changed, and only its checks while every such file is as it was.
	resumed := filepath.Join(t.TempDir(), cycleStateFile)
	for _, step := range []struct {
		consumed string
		want     int
	}{{"files as bound", 4}, {"files as bound", 2}, {"a file changed", 4}} {
		ran := 0
		if err := measurePhases(resumed, "commit|store|"+digest(`,"consumed":"`+step.consumed+`"`), io.Discard, nil, nil, func([]string, ...string) error {
			ran++
			return nil
		}); err != nil || ran != step.want {
			t.Errorf("a cycle over %s ran %d phases (%v), want %d", step.consumed, ran, err, step.want)
		}
	}

	state := filepath.Join(t.TempDir(), cycleStateFile)
	var told [][]string
	if err := measurePhases(state, "commit|store|obligations", io.Discard, nil, nil, func(environment []string, _ ...string) error {
		told = append(told, environment)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	want := clioptions.CycleCreditEnvironment + "=" + filepath.Join(filepath.Dir(state), cycleCreditFile)
	if len(told) != 4 {
		t.Fatalf("the cycle ran %d phases, want four", len(told))
	}
	for index, environment := range told {
		if !slices.Contains(environment, want) {
			t.Errorf("phase %d was told %q, want the credit file %q among them", index, environment, want)
		}
	}
}

// TestObligationsDigestIgnoresWhatARunChanges holds the key's obligations to
// the cells the plan names and why, not to the cost a run measured or the
// record it replays, which every cell the cycle runs changes.
func TestObligationsDigestIgnoresWhatARunChanges(t *testing.T) {
	t.Parallel()
	digest := func(plan string) string {
		t.Helper()
		value, err := obligationsDigest([]byte(plan))
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	base := digest(`{"run":[{"model":"m","kind":"inference","validation":"guard","recipe":"r1","cost_ns":5,"replay":"x"}],"background":[]}`)
	if got := digest(`{"run":[{"model":"m","kind":"inference","validation":"guard","recipe":"r1","cost_ns":900,"replay":"y"}],"background":[]}`); got != base {
		t.Error("a run's own cost and replay record changed the obligations")
	}
	for name, plan := range map[string]string{
		"another recipe":  `{"run":[{"model":"m","kind":"inference","validation":"guard","recipe":"r2"}],"background":[]}`,
		"another cell":    `{"run":[{"model":"m","kind":"inference","validation":"guard","recipe":"r1"},{"model":"n","kind":"inference","validation":"guard","recipe":"r1"}],"background":[]}`,
		"another bucket":  `{"run":[],"background":[{"model":"m","kind":"inference","validation":"guard","recipe":"r1"}]}`,
		"another promote": `{"run":[{"model":"m","kind":"inference","validation":"guard","recipe":"r1","promotion":"background: x"}],"background":[]}`,
	} {
		if digest(plan) == base {
			t.Errorf("%s left the obligations as they were", name)
		}
	}
}

// TestTheGapsAreFiledOnceTheScorecardPublishes holds a cycle to filing what its scorecard found as
// soon as the scorecard is published; the hook runs again when an interrupted run's publication is
// credited, and never when the scorecard did not publish.
func TestTheGapsAreFiledOnceTheScorecardPublishes(t *testing.T) {
	t.Parallel()
	var order []string
	failing := ""
	validate := func(_ []string, arguments ...string) error {
		phase := strings.Join(arguments, " ")
		order = append(order, phase)
		if phase == failing {
			return errors.New("ended")
		}
		return nil
	}
	refuse := false
	filed := func() error {
		order = append(order, "filed")
		if refuse {
			return errors.New("not filed")
		}
		return nil
	}
	const preflight, lanes, cells, publish = "-run -preflight", "-run -lanes", "-run -cells", "-scorecard -publish"
	for _, step := range []struct {
		name, key, fail string
		want            []string
	}{
		{"a cycle", "commit-a|store|obligations", "", []string{preflight, lanes, cells, publish, "filed"}},
		{"a scorecard that did not publish", "commit-b|store|obligations", publish, []string{preflight, lanes, cells, publish}},
	} {
		order, failing = nil, step.fail
		_ = measurePhases(filepath.Join(t.TempDir(), cycleStateFile), step.key, io.Discard, filed, nil, validate)
		if !slices.Equal(order, step.want) {
			t.Errorf("%s: ran %q, want %q", step.name, order, step.want)
		}
	}
	state := filepath.Join(t.TempDir(), cycleStateFile)
	failing, order, refuse = "", nil, true
	_ = measurePhases(state, "commit-c|store|obligations", io.Discard, filed, nil, validate)
	order, refuse = nil, false
	_ = measurePhases(state, "commit-c|store|obligations", io.Discard, filed, nil, validate)
	if want := []string{preflight, lanes, "filed"}; !slices.Equal(order, want) {
		t.Errorf("the interrupted cycle's credited publication: ran %q, want %q", order, want)
	}
}
