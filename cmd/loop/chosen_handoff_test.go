package main

import (
	"errors"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestPublicationReadsTheCellsPhasesChoice holds the run's record of the cells its cells phase chose
// (plan.MeasurementRun.Chosen) to the publication that reads it (run-record-chosen, SQA-527): the loop's own writes of
// the record keep the choice the cells phase's validate wrote, an empty choice as empty, and a cycle begun again after
// its cells phase earned credit gives its publication the same choice without measuring a cell again. At c87ae04a3 the
// loop rewrote the record on the phase's exit without it, so the publication read none and fell back to the list.
func TestPublicationReadsTheCellsPhasesChoice(t *testing.T) {
	t.Parallel()
	for _, chose := range [][]string{{}, {"model:sha256:qwen inference"}} {
		root, state := t.TempDir(), filepath.Join(t.TempDir(), cycleStateFile)
		var seen [][]string
		cells := 0
		validate := func(environment []string, arguments ...string) error {
			var record string
			for _, variable := range environment {
				if path, found := strings.CutPrefix(variable, clioptions.MeasurementRunEnvironment+"="); found {
					record = path
				}
			}
			switch strings.Join(arguments, " ") {
			case "-run -cells -reached":
				cells++
				return plan.RecordChosen(record, chose)
			case "-scorecard -publish":
				var run plan.MeasurementRun
				testcheck.NoError(t, jsonfile.Decode(record, &run))
				seen = append(seen, run.Chosen)
				if len(seen) == 1 {
					return errors.New("the first publication is cut short")
				}
			}
			return nil
		}
		for range 2 {
			run := &runRecorder{root: root}
			run.begin("measured")
			_ = measurePhases(state, "commit|store|obligations", io.Discard, nil, run, validate)
		}
		if cells != 1 || len(seen) != 2 {
			t.Fatalf("choosing %q, the cells phase ran %d times and the publication %d; want once and twice, the cells credited when the cycle began again", chose, cells, len(seen))
		}
		for attempt, read := range seen {
			if read == nil || !slices.Equal(read, chose) {
				t.Errorf("choosing %q, publication %d read the choice %q (nil %t); want the cells phase's", chose, attempt+1, read, read == nil)
			}
		}
	}
}
