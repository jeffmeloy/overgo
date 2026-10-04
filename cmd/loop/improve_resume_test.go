package main

import (
	"errors"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestInterruptedCycleResumesAtItsFirstUnpassedPhase holds a cycle to going on
// from where it stopped for the same key: the cells and the publication that
// passed are not run again when a run ended in the lanes, the lanes themselves
// always run (go test's cache replays what passed), a cell failure runs the
// cells and the publication again, and another key, another commit, store or
// obligation, starts over.
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
		err := measurePhases(state, key, validate)
		return ran, err
	}
	const cells, publish, lanes = "-run -cells", "-scorecard -publish", "-run -lanes"
	for _, step := range []struct {
		name, key, fail string
		want            []string
		wantErr         bool
	}{
		{"a cycle that ends in its lanes", "commit-a|store|obligations", lanes, []string{cells, publish, lanes}, true},
		{"the next run goes on at the lanes", "commit-a|store|obligations", "", []string{lanes}, false},
		{"another store starts over", "commit-a|other-store|obligations", cells, []string{cells, publish, lanes}, true},
		{"so does an obligation that changed", "commit-a|other-store|activated", "", []string{cells, publish, lanes}, false},
		{"another commit starts over", "commit-b|other-store|activated", "", []string{cells, publish, lanes}, false},
	} {
		got, err := cycle(step.key, step.fail)
		if !slices.Equal(got, step.want) || (err != nil) != step.wantErr {
			t.Errorf("%s: ran %q with error %v, want %q (error %t)", step.name, got, err, step.want, step.wantErr)
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
