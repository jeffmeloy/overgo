package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/scorecard"
)

// evaluationFailure opens what a run's record names a failed benchmark by, its model, family and the protocol its cell
// was bound to after it (none where it named none), which the run's report prints beside its failed tests.
const evaluationFailure = "evaluation "

// cellFailure opens what a run's record names any other broken cell by, what it measures and its cause's first line
// after it.
const cellFailure = "cell "

// noteFailedCell tells the measurement run under way that a cell broke, in the phase running, the moment it breaks,
// so the run's report names it while the other cells run. A benchmark is named by its model, family and protocol, for
// the run's publication to read, so one protocol's failure hides no other's reading (failed-protocol, SQA-530): the scorecard's store holds the reading from before, and only this says that this run did not
// take one. Any other cell is named by subject, what it measures, and the first line of its cause: Flash-Next's broken
// projection proof at 3a7b3877d reached only the run's terminal error, its cells phase ending 2 h 43 min later with no
// failed entry (proof-failure-recorded, SQA-528).
func noteFailedCell(output io.Writer, cell ModelValidation, subject string, broke error) {
	path := os.Getenv(clioptions.MeasurementRunEnvironment)
	if path == "" {
		return
	}
	cause, _, _ := strings.Cut(broke.Error(), "\n")
	failed := cellFailure + subject + ": " + cause
	if benchmark(cell) {
		failed = strings.TrimSpace(evaluationFailure + cell.Model.String() + " " + cell.Validation + " " + cell.Protocol)
	}
	if err := plan.RecordFailure(path, failed); err != nil {
		fmt.Fprintf(output, "validate -run: the run record %s does not name %s: %v\n", path, subject, err)
	}
}

// withoutFailedEvaluations is rows with each quality the measurement run under way failed to evaluate said to be
// unmeasured (scorecard.Record.NotEvaluated), and rows as they are outside a run.
func withoutFailedEvaluations(rows []scorecard.Row) ([]scorecard.Row, error) {
	path := os.Getenv(clioptions.MeasurementRunEnvironment)
	if path == "" {
		return rows, nil
	}
	failed, err := failedEvaluations(path)
	if err != nil {
		return nil, err
	}
	return scorecard.Record{Rows: rows}.NotEvaluated(failed).Rows, nil
}

// failedEvaluations reads from the run record at path what says whether a model's evaluation on a suite under a
// contract's protocol failed this run: a failure names its protocol, and one named by family alone, as a record from
// before failed-protocol names it, failed every protocol's.
func failedEvaluations(path string) (func(model artifact.ID, contract scorecard.Contract) bool, error) {
	var run plan.MeasurementRun
	if err := jsonfile.Decode(path, &run); err != nil {
		return nil, fmt.Errorf("validate: the run record %s: %w", path, err)
	}
	type evaluated struct{ model, family, protocol string }
	var failed []evaluated
	for _, phase := range run.Phases {
		for _, name := range phase.Failed {
			if named, evaluation := strings.CutPrefix(name, evaluationFailure); evaluation {
				if model, family, found := strings.Cut(named, " "); found {
					family, protocol, _ := strings.Cut(family, " ")
					failed = append(failed, evaluated{model, family, protocol})
				}
			}
		}
	}
	return func(model artifact.ID, contract scorecard.Contract) bool {
		for _, evaluation := range failed {
			if evaluation.model == model.String() && ofFamily(contract.Suite, evaluation.family) && (evaluation.protocol == "" || evaluation.protocol == contract.Protocol) {
				return true
			}
		}
		return false
	}, nil
}
