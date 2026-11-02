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

// evaluationFailure opens what a run's record names a failed benchmark by, its model and family after it, which the
// run's report prints beside its failed tests.
const evaluationFailure = "evaluation "

// noteFailedEvaluation tells the measurement run under way that a benchmark failed, for its publication to read: the
// scorecard's store holds the reading from before, and only this says that this run did not take one.
func noteFailedEvaluation(output io.Writer, cell ModelValidation) {
	path := os.Getenv(clioptions.MeasurementRunEnvironment)
	if path == "" {
		return
	}
	if err := plan.RecordFailure(path, evaluationFailure+cell.Model.String()+" "+cell.Validation); err != nil {
		fmt.Fprintf(output, "validate -run: the run record %s does not name %s %s: %v\n", path, cell.Model, cell.Validation, err)
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

// failedEvaluations reads from the run record at path what says whether a model's evaluation on a suite failed this run.
func failedEvaluations(path string) (func(model artifact.ID, suite string) bool, error) {
	var run plan.MeasurementRun
	if err := jsonfile.Decode(path, &run); err != nil {
		return nil, fmt.Errorf("validate: the run record %s: %w", path, err)
	}
	type evaluated struct{ model, family string }
	var failed []evaluated
	for _, phase := range run.Phases {
		for _, name := range phase.Failed {
			if named, evaluation := strings.CutPrefix(name, evaluationFailure); evaluation {
				if model, family, found := strings.Cut(named, " "); found {
					failed = append(failed, evaluated{model, family})
				}
			}
		}
	}
	return func(model artifact.ID, suite string) bool {
		for _, evaluation := range failed {
			if evaluation.model == model.String() && ofFamily(suite, evaluation.family) {
				return true
			}
		}
		return false
	}, nil
}
