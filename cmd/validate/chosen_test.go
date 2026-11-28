package main

import (
	"bytes"
	"context"
	"io"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/landing"
	"overgo/internal/overgodbtest"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestACellTheRunDidNotChooseOwesNoReading holds the scorecard's mark of the readings a run owes to the cells the run
// chose (carried-reached, SQA-527): a change-selected run leaves a listed model's cells it does not reach unchosen, as
// MOSS's image cell after a change confined to cmd/server. Its cells phase keeps its choice in the run's record, and
// the run's publication marks MOSS's row unchosen and the chosen Qwen3.5-4B's not, both on the measured models list.
func TestACellTheRunDidNotChooseOwesNoReading(t *testing.T) {
	// Serial: it names a run record and a credit file of its own in the process environment.
	directory := t.TempDir()
	t.Setenv(clioptions.CycleCreditEnvironment, filepath.Join(directory, "credit.json"))
	record := filepath.Join(directory, "run.json")
	began := time.Date(2026, 11, 28, 3, 32, 0, 0, time.UTC)
	testcheck.NoError(t, jsonfile.Write(record, plan.MeasurementRun{Started: began, Phases: []plan.MeasurementPhase{{Name: "cells", Started: began}}}, clioptions.PrivateFileMode))
	t.Setenv(clioptions.MeasurementRunEnvironment, record)

	cell := func(name string, task recipe.Task, validation string) Selected {
		return Selected{ModelValidation: ModelValidation{
			Model: testutil.ArtifactID(t, artifact.KindModel, name), Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name),
			ModelName: name, Kind: task, Validation: validation, Location: "C:/models/" + name,
		}}
	}
	qwen, moss := cell("Qwen3.5-4B", recipe.TaskInference, servingValidation), cell("MOSS", recipe.TaskImageGen, servingValidation)
	measure := func(context.Context, io.Writer, func() error) (landing.Reading, error) { return landing.Reading{}, nil }
	var out bytes.Buffer
	chosen := Plan{Run: []Selected{qwen}, NotChosen: []Selected{moss}}
	testcheck.NoError(t, executeMeasured(t.Context(), &out, testutil.RepoRoot(t), "C:/stores/selected", nil, chosen, scopeCells, measure))

	root := t.TempDir()
	testutil.WriteTextFile(t, root, "docs/verification/measured_models.json", `{"models": [{"name": "MOSS"}, {"name": "Qwen3.5-4B"}]}`, 0o600)
	rows := []scorecard.Row{{Model: moss.Model, Name: "MOSS", Task: recipe.TaskImageGen}, {Model: qwen.Model, Name: "Qwen3.5-4B", Task: recipe.TaskInference}}
	marked := testcheck.Must(markUnchosen(t.Context(), overgodbtest.Create(t, t.TempDir()), root, rows))(t)
	for _, row := range marked {
		if want := row.Model == moss.Model; row.Unchosen != want {
			t.Errorf("%s %s is marked unchosen %t, want %t: the run chose Qwen's cell alone\n%s", row.Name, row.Task, row.Unchosen, want, out.String())
		}
	}
}
