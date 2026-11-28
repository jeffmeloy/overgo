package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/landing"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestABrokenProofIsRecordedInItsPhase holds a broken recipe proof to the phase it broke in (proof-failure-recorded,
// SQA-528): the run's record names the cell, what it measures and the first line of its cause the moment it breaks, so
// the run's report names it while the other cells run. Flash-Next's projection proof broke at 3a7b3877d, its exact suite
// refusing image-1-bottom's " D", and the cells phase ended 2 h 43 min later with no failed entry, the cause only in the
// run's terminal error.
func TestABrokenProofIsRecordedInItsPhase(t *testing.T) {
	// Serial: it names a run record and a credit file of its own in the process environment.
	directory := t.TempDir()
	t.Setenv(clioptions.CycleCreditEnvironment, filepath.Join(directory, "credit.json"))
	record := filepath.Join(directory, "run.json")
	began := time.Date(2026, 11, 28, 3, 32, 0, 0, time.UTC)
	testcheck.NoError(t, jsonfile.Write(record, plan.MeasurementRun{Started: began, Phases: []plan.MeasurementPhase{{Name: "cells", Started: began}}}, clioptions.PrivateFileMode))
	t.Setenv(clioptions.MeasurementRunEnvironment, record)

	proof := Selected{ModelValidation: ModelValidation{
		Model: testutil.ArtifactID(t, artifact.KindModel, "flash"), Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "flash"),
		ModelName: "Qwen3.8 Flash Next", Kind: recipe.TaskProjection, Validation: projectionProof,
		Grounding: "docs/verification/flash-next-exact.json", Location: "C:/models/flash.gguf", Projector: "C:/models/mmproj.gguf",
	}}
	cause := errors.New("exact suite refused image-1-bottom: generated \" D\", want \"D\"\nthe rest of its report")
	measure := func(context.Context, io.Writer, func() error) (landing.Reading, error) {
		return landing.Reading{}, cause
	}
	var out bytes.Buffer
	if err := executeMeasured(t.Context(), &out, testutil.RepoRoot(t), "C:/stores/selected", nil, Plan{Run: []Selected{proof}}, scopeCells, measure); err == nil {
		t.Fatalf("a broken proof ended its run as passed\n%s", out.String())
	}
	var run plan.MeasurementRun
	testcheck.NoError(t, jsonfile.Decode(record, &run))
	failed := run.Phases[0].Failed
	if len(failed) != 1 || !strings.Contains(failed[0], projectionProof) || !strings.Contains(failed[0], "Qwen3.8 Flash Next") ||
		!strings.Contains(failed[0], "image-1-bottom") || strings.Contains(failed[0], "the rest of its report") {
		t.Fatalf("the phase records %q, want the broken proof by what it measures and its cause's first line", failed)
	}
}
