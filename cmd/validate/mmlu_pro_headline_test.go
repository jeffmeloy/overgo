package main

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestMMLUProIsHeadedByTheLeaderboardReading holds the validator to the owner's direction of 2026-11-10: MMLU-Pro is
// read by lm-eval's leaderboard_mmlu_pro reading, raw completion after five validation demonstrations, which heads its
// cell, and beside it through the chat template, never one in the other's place. An inference activation's MMLU-Pro
// is two cells, the leaderboard reading first; a baseline that compares the chat-template reading takes that cell and
// leaves the leaderboard cell its own; the leaderboard cell runs evaluate under its default, raw completion, and the
// chat cell names the chat template; and the other families keep one cell naming no protocol.
func TestMMLUProIsHeadedByTheLeaderboardReading(t *testing.T) {
	t.Parallel()
	raw, chat := evaluation.PromptingRawCompletion.Label(), evaluation.PromptingChatTemplate.Label()
	model, activation := testutil.ArtifactID(t, artifact.KindModel, "read"), testutil.ArtifactID(t, artifact.KindRecipe, "read/inference")
	var cells []ModelValidation
	for _, validation := range kindValidation(recipe.TaskInference) {
		cells = append(cells, validationCells(ModelValidation{
			Model: model, ModelName: "read", Location: "C:/models/read.gguf", Kind: recipe.TaskInference, Recipe: activation, Validation: validation,
		})...)
	}
	var mmluPro []string
	for _, cell := range cells {
		switch {
		case cell.Validation == "mmlu-pro":
			mmluPro = append(mmluPro, cell.Protocol)
		case cell.Protocol != "":
			t.Errorf("the %s cell names protocol %q; only MMLU-Pro is read under more than one", cell.Validation, cell.Protocol)
		}
	}
	if !slices.Equal(mmluPro, []string{raw, chat}) {
		t.Fatalf("MMLU-Pro's cells read under %q, want the leaderboard reading, %s, first and the chat template beside it", mmluPro, raw)
	}

	source := testutil.ArtifactID(t, artifact.KindEvidence, "baseline")
	baseline := &scorecard.Record{Rows: []scorecard.Row{{Model: model, Name: "read", Task: recipe.TaskInference, Recipe: activation, Cells: map[metric.Name]scorecard.Cell{
		metric.Quality: {Value: 0.5, Source: source, Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "store/mmlu-pro", Protocol: chat}},
	}}}}
	compared := compareContracts(slices.Clone(cells), baseline)
	var after []string
	for _, cell := range compared {
		if cell.Validation == "mmlu-pro" {
			after = append(after, cell.Protocol)
		}
	}
	if !slices.Equal(after, []string{raw, chat}) {
		t.Fatalf("with the chat-template reading compared, MMLU-Pro's cells read under %q; want the leaderboard cell kept beside it", after)
	}
	for _, cell := range compared {
		if cell.Validation != "mmlu-pro" {
			continue
		}
		_, args, err := reacquireCommand(cell, "C:/stores/read")
		testcheck.NoError(t, err)
		if named := slices.Contains(args, "-protocol"); named != (cell.Protocol == chat) {
			t.Errorf("MMLU-Pro's %s cell runs %v; want the chat template named for the chat cell alone, the leaderboard reading evaluate's default", cell.Protocol, args)
		}
	}
}
