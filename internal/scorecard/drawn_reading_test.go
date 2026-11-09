package scorecard

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestDrawnTextScoresTheCharactersReadBack holds an image model's drawn-text cell to the character score its reading
// recorded: the share of the asked characters the reader read back, stated beside the judge's share, which stays the
// cell's value; a reading no reader read states none.
func TestDrawnTextScoresTheCharactersReadBack(t *testing.T) {
	t.Parallel()
	summary := evaluation.EvalSummary{
		Suite: "drawn-text", Prompting: "judged", Record: testutil.ArtifactID(t, artifact.KindEvaluation, "characters"),
		Metrics:    map[string]float64{"accuracy": 1, evaluation.SoftMetric: 0.97, evaluation.CharacterMetric: 0.9901},
		Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize, evaluation.SoftMetric: runrecord.DirectionMaximize, evaluation.CharacterMetric: runrecord.DirectionMaximize},
	}
	suite := map[string]evaluation.SuiteDescriptor{"drawn-text": {Kind: "generated-answer", Source: "drawn-text", Cases: 10}}
	cell, _ := qualityCell(recipe.TaskImageGen, []evaluation.EvalSummary{summary}, suite)
	if cell.Value != 1 || cell.Characters == nil || *cell.Characters != 0.9901 || !strings.Contains(cell.QualityText(), "0.9901 of the asked characters read back") {
		t.Fatalf("a read drawn-text reading reads %v with characters %v as %q, want 1 with 0.9901 beside it", cell.Value, cell.Characters, cell.QualityText())
	}
	t.Logf("the drawn-text cell reads %q", cell.QualityText())
	delete(summary.Metrics, evaluation.CharacterMetric)
	if unread, _ := qualityCell(recipe.TaskImageGen, []evaluation.EvalSummary{summary}, suite); unread.Characters != nil || strings.Contains(unread.QualityText(), "characters") {
		t.Fatalf("a reading no reader read reads %q", unread.QualityText())
	}
}
