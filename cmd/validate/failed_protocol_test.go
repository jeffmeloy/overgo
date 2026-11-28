package main

import (
	"bytes"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAFailedProtocolHidesNoOtherProtocolsReading holds a failed benchmark to the readings of its own protocol
// (failed-protocol, SQA-530): Flash-Next's MMLU-Pro is read under raw completion and through the chat template, each
// of 1,209 cases, and a run whose chat-template evaluation broke publishes that reading as not measured while its raw
// headline (68.57%) and its historical reading under raw completion without demonstrations stand. A failure recorded
// by family alone, as runs before it recorded one, still says every reading of the family was not measured.
func TestAFailedProtocolHidesNoOtherProtocolsReading(t *testing.T) {
	// Serial: it names a run record of its own in the process environment.
	raw, chat := evaluation.PromptingRawCompletion.Label(), evaluation.PromptingChatTemplate.Label()
	historical := raw + ", without demonstrations"
	model := testutil.ArtifactID(t, artifact.KindModel, "flash")
	reading := func(name string, value float64, protocol string) scorecard.Cell {
		return scorecard.Cell{Value: value, Source: testutil.ArtifactID(t, artifact.KindEvaluation, name),
			Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "store/mmlu-pro", Protocol: protocol}}
	}
	rows := []scorecard.Row{{
		Model: model, Name: "Qwen3.8 Flash Next", Task: recipe.TaskInference,
		Cells:   map[metric.Name]scorecard.Cell{metric.Quality: reading("raw", 0.6857, raw)},
		Heldout: []scorecard.Cell{reading("chat", 0.6584, chat), reading("historical", 0.6261, historical)},
	}}
	record := func() string {
		path := filepath.Join(t.TempDir(), "run.json")
		began := time.Date(2026, 11, 28, 3, 32, 0, 0, time.UTC)
		testcheck.NoError(t, jsonfile.Write(path, plan.MeasurementRun{Started: began, Phases: []plan.MeasurementPhase{{Name: "cells", Started: began}}}, clioptions.PrivateFileMode))
		t.Setenv(clioptions.MeasurementRunEnvironment, path)
		return path
	}
	record()
	var output bytes.Buffer
	broken := ModelValidation{Model: model, Kind: recipe.TaskInference, Validation: "mmlu-pro", Protocol: chat}
	noteFailedCell(&output, broken, "Qwen3.8 Flash Next mmlu-pro", errors.New("the chat-template worker ended"))
	published := testcheck.Must(withoutFailedEvaluations(rows))(t)[0]
	if headline := published.Cells[metric.Quality]; !headline.Measured() || headline.Contract.Protocol != raw {
		t.Errorf("the raw headline reads %+v after the chat-template evaluation alone broke; want its 0.6857 standing", headline)
	}
	if len(published.Heldout) != 1 || published.Heldout[0].Contract.Protocol != historical {
		t.Errorf("the held-out readings are %+v; want the historical reading alone, the broken chat-template one not measured", published.Heldout)
	}

	// A run's record from before the protocol was kept names the family alone, and says all of it was not measured.
	path := record()
	testcheck.NoError(t, plan.RecordFailure(path, evaluationFailure+model.String()+" mmlu-pro"))
	if older := testcheck.Must(withoutFailedEvaluations(rows))(t)[0]; older.Cells[metric.Quality].Measured() || len(older.Heldout) != 0 {
		t.Errorf("a failure named by family alone leaves %+v and %+v; want every reading of the family not measured", older.Cells[metric.Quality], older.Heldout)
	}
}
