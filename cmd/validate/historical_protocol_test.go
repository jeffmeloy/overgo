package main

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAHistoricalProtocolIsReadNotRun holds the cells a baseline's held-out contracts give a model to the protocols
// the evaluator runs (historical-protocol, SQA-529): Flash-Next's scorecard keeps an MMLU-Pro reading under raw
// completion without demonstrations, a protocol evaluate no longer takes, beside its leaderboard and chat-template
// readings. Its MMLU-Pro is the two cells it is read under, each run with arguments of its own, and the historical
// reading schedules none: on 2026-11-25 it was a third cell run with the leaderboard cell's arguments.
func TestAHistoricalProtocolIsReadNotRun(t *testing.T) {
	t.Parallel()
	raw, chat := evaluation.PromptingRawCompletion.Label(), evaluation.PromptingChatTemplate.Label()
	historical := raw + ", without demonstrations"
	model, activation := testutil.ArtifactID(t, artifact.KindModel, "flash"), testutil.ArtifactID(t, artifact.KindRecipe, "flash/inference")
	var cells []ModelValidation
	for _, validation := range kindValidation(recipe.TaskInference) {
		cells = append(cells, validationCells(ModelValidation{
			Model: model, ModelName: "flash", Location: "C:/models/flash.gguf", Kind: recipe.TaskInference, Recipe: activation, Validation: validation,
		})...)
	}
	reading := func(name string, value float64, protocol string) scorecard.Cell {
		return scorecard.Cell{Value: value, Source: testutil.ArtifactID(t, artifact.KindEvidence, name),
			Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "store/mmlu-pro", Protocol: protocol}}
	}
	baseline := &scorecard.Record{Rows: []scorecard.Row{{
		Model: model, Name: "Qwen3.8 Flash Next", Task: recipe.TaskInference, Recipe: activation,
		Cells:   map[metric.Name]scorecard.Cell{metric.Quality: reading("leaderboard", 0.686, raw)},
		Heldout: []scorecard.Cell{reading("chat", 0.658, chat), reading("historical", 0.626, historical)},
	}}}
	var protocols []string
	var commands []string
	for _, cell := range compareContracts(slices.Clone(cells), baseline) {
		if cell.Validation != "mmlu-pro" {
			continue
		}
		protocols = append(protocols, cell.Protocol)
		_, args, err := reacquireCommand(cell, "C:/stores/flash")
		testcheck.NoError(t, err)
		commands = append(commands, strings.Join(args, " "))
	}
	if !slices.Equal(protocols, []string{raw, chat}) {
		t.Fatalf("Flash-Next's MMLU-Pro is scheduled under %q, want the leaderboard reading and the chat template, the historical %q read alone", protocols, historical)
	}
	if commands[0] == commands[1] {
		t.Errorf("both MMLU-Pro cells run %q; each protocol's cell runs its own", commands[0])
	}
}
