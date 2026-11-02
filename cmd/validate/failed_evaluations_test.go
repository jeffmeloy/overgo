package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestAFailedBenchmarkReachesItsPublication: a benchmark that fails is named in the run's record as it fails, and the
// publication of that run reads which model's suite did not answer, and nothing about any other.
func TestAFailedBenchmarkReachesItsPublication(t *testing.T) {
	path := filepath.Join(t.TempDir(), "run.json")
	running := plan.MeasurementRun{Started: time.Now(), Phases: []plan.MeasurementPhase{{Name: "cells", Started: time.Now()}}}
	if err := jsonfile.Write(path, running, clioptions.OutputFileMode); err != nil {
		t.Fatal(err)
	}
	t.Setenv(clioptions.MeasurementRunEnvironment, path)
	model := testutil.ArtifactID(t, artifact.KindModel, "failed model")
	other := testutil.ArtifactID(t, artifact.KindModel, "other model")
	var output bytes.Buffer
	noteFailedEvaluation(&output, ModelValidation{Model: model, Validation: "mmlu-pro"})
	if output.Len() != 0 {
		t.Fatalf("recording the failure said %q", output.String())
	}
	failed, err := failedEvaluations(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, read := range []struct {
		model artifact.ID
		suite string
		want  bool
	}{
		{model, "store/mmlu-pro", true},
		{model, "store/bbh", false},
		{other, "store/mmlu-pro", false},
	} {
		if got := failed(read.model, read.suite); got != read.want {
			t.Errorf("%s on %s failed = %t, want %t", read.model, read.suite, got, read.want)
		}
	}

	// The scorecard of the run reads the gap where the reading from before would read as holding.
	reading := scorecard.Cell{Value: 0.5, Source: testutil.ArtifactID(t, artifact.KindEvaluation, "before"), Contract: scorecard.Contract{Suite: "store/mmlu-pro"}}
	rows, err := withoutFailedEvaluations([]scorecard.Row{{Model: model, Cells: map[metric.Name]scorecard.Cell{metric.Quality: reading}}})
	if err != nil {
		t.Fatal(err)
	}
	if cell := rows[0].Cells[metric.Quality]; cell.Measured() || !strings.Contains(cell.Reason, "its evaluation on store/mmlu-pro failed") {
		t.Fatalf("the failed evaluation's quality reads %+v", cell)
	}

	// The scorecard reads the record of the run: one it cannot read refuses, rather than print what may have failed as holding.
	t.Setenv(clioptions.MeasurementRunEnvironment, filepath.Join(t.TempDir(), "no-run.json"))
	if err := runArgs(t.Context(), []string{"-scorecard", "-repo", scorecardFixture(t)}, &output); err == nil || !strings.Contains(err.Error(), "the run record") {
		t.Fatalf("a scorecard under a run with no record = %v, want it refused", err)
	}
}
