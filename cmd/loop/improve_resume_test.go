package main

import (
	"errors"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestInterruptedCycleResumesAtItsFirstUnpassedPhase holds a cycle to going on
// from where it stopped: the cells and the publication that passed are not run
// again at the same commit when a run ended in the lanes, a cell failure runs
// the cells and the publication again, and another commit starts over.
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
	cycle := func(commit, failPhase string) ([]string, error) {
		ran, failing = nil, failPhase
		err := measurePhases(state, commit, validate)
		return ran, err
	}
	const cells, publish, lanes = "-run -cells", "-scorecard -publish", "-run -lanes"
	for _, step := range []struct {
		name, commit, fail string
		want               []string
		wantErr            bool
	}{
		{"a cycle that ends in its lanes", "a", lanes, []string{cells, publish, lanes}, true},
		{"the next run goes on at the lanes", "a", "", []string{lanes}, false},
		{"a finished cycle has nothing left at its commit", "a", "", nil, false},
		{"another commit starts over", "b", cells, []string{cells, publish, lanes}, true},
		{"failed cells run again, and publish again after them", "b", "", []string{cells, publish}, false},
	} {
		got, err := cycle(step.commit, step.fail)
		if !slices.Equal(got, step.want) || (err != nil) != step.wantErr {
			t.Errorf("%s: ran %q with error %v, want %q (error %t)", step.name, got, err, step.want, step.wantErr)
		}
	}
}
