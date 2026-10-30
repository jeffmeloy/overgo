package plan

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
)

// measurementRunFile is the record a measurement run (cmd/loop -improve, or the
// cycle the loop runs when the plan empties) keeps of itself in the process
// state of the checkout it runs from. The session hooks print it, so a run's end
// and its times are read from it, never remembered or reckoned: a stop once
// said a run launched at 02:21 started at 05:35, and on 2026-10-29 a detached
// run ended unnoticed for five hours.
const measurementRunFile = "measurement_run.json"

// MeasurementRun is a run's record of itself: the commit it measures, its
// process, when it began, each phase as it runs, and when it ended with the
// rows it filed.
type MeasurementRun struct {
	Commit  string             `json:"commit"`
	Process int                `json:"process"`
	Started time.Time          `json:"started"`
	Phases  []MeasurementPhase `json:"phases"`
	Ended   time.Time          `json:"ended,omitzero"`
	Filed   int                `json:"filed,omitzero"`
}

// MeasurementPhase is one phase of a run: when it began, when it is expected to
// end by the wall the same phase took in the run before (zero where none took
// it), when it ended and whether it passed, or that an interrupted run's credit
// stood for it.
type MeasurementPhase struct {
	Name     string    `json:"name"`
	Started  time.Time `json:"started"`
	Expected time.Time `json:"expected,omitzero"`
	Ended    time.Time `json:"ended,omitzero"`
	Passed   bool      `json:"passed,omitzero"`
	Credited bool      `json:"credited,omitzero"`
}

// MeasurementRunPath is where the run record of the checkout at root is kept.
func MeasurementRunPath(root string) string {
	return filepath.Join(root, processlock.StateDirectory, measurementRunFile)
}

// ReadMeasurementRun reads the newest run's record of the checkout at root;
// false when none is kept or it cannot be read.
func ReadMeasurementRun(root string) (MeasurementRun, bool) {
	var run MeasurementRun
	return run, jsonfile.Decode(MeasurementRunPath(root), &run) == nil && !run.Started.IsZero()
}

// Wall is how long the phase named took in the run, and whether it ended there.
func (run MeasurementRun) Wall(name string) (time.Duration, bool) {
	for _, phase := range run.Phases {
		if phase.Name == name && !phase.Ended.IsZero() && !phase.Credited {
			return phase.Ended.Sub(phase.Started), true
		}
	}
	return 0, false
}

// Report is the run as the hooks print it: when it began, each phase with its
// verdict and wall or, running, since when and its expected end, and its end,
// or, where its process is gone with no end recorded, that it was cut short.
func (run MeasurementRun) Report() string {
	at := func(moment time.Time) string { return moment.Local().Format("Jan 2 15:04 MST") }
	var b strings.Builder
	fmt.Fprintf(&b, "measurement run at %.12s, process %d, began %s", run.Commit, run.Process, at(run.Started))
	for _, phase := range run.Phases {
		switch {
		case phase.Credited:
			fmt.Fprintf(&b, "; %s credited from the interrupted run", phase.Name)
		case !phase.Ended.IsZero():
			verdict := "passed"
			if !phase.Passed {
				verdict = "did not pass"
			}
			fmt.Fprintf(&b, "; %s %s in %s", phase.Name, verdict, phase.Ended.Sub(phase.Started).Round(time.Minute))
		case phase.Expected.IsZero():
			fmt.Fprintf(&b, "; %s running since %s, no earlier run timed it", phase.Name, at(phase.Started))
		default:
			fmt.Fprintf(&b, "; %s running since %s, expected to end %s", phase.Name, at(phase.Started), at(phase.Expected))
		}
	}
	switch {
	case !run.Ended.IsZero():
		fmt.Fprintf(&b, "; ended %s, after %s, filing %d rows", at(run.Ended), run.Ended.Sub(run.Started).Round(time.Minute), run.Filed)
	case !processcontrol.ProcessAlive(run.Process):
		b.WriteString("; its process is gone with no end recorded: the run was cut short")
	default:
		b.WriteString("; running: plan-only commits, no device work, no push until it ends")
	}
	return b.String()
}
