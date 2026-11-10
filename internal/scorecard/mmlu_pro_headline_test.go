package scorecard

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestTheLeaderboardReadingHeadsMMLUPro holds MMLU-Pro's quality cell to its headline reading: of a model's readings
// of it, the leaderboard reading, raw completion after the validation split's five demonstrations, heads the cell
// though the chat-template reading is newer, and the chat-template reading stands beside it; a likelihood reading
// asked without the demonstrations, labelled apart, heads nothing; and with no leaderboard reading the newest heads,
// as before.
func TestTheLeaderboardReadingHeadsMMLUPro(t *testing.T) {
	t.Parallel()
	raw, chat := evaluation.PromptingRawCompletion.Label(), evaluation.PromptingChatTemplate.Label()
	reading := func(suite, protocol string, accuracy float64) evaluation.EvalSummary {
		return evaluation.EvalSummary{
			Suite: suite, Prompting: protocol, Record: testutil.ArtifactID(t, artifact.KindEvaluation, suite+protocol),
			Metrics: map[string]float64{"accuracy": accuracy}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize},
		}
	}
	descriptors := map[string]evaluation.SuiteDescriptor{
		"leaderboard": {Kind: "grouped-choice", Source: "store/mmlu-pro", Cases: 1203},
		"chat":        {Kind: "grouped-choice", Source: "store/mmlu-pro", Cases: 1203},
		"alone":       {Kind: "grouped-choice", Source: "store/mmlu-pro", Cases: 1203},
	}
	leaderboard, asked := reading("leaderboard", raw, 0.19), reading("chat", chat, 0.31)
	alone := reading("alone", raw+", without demonstrations", 0.12)
	// The newest reading comes first in the summaries a recipe's evidence lists.
	cell, beside := qualityCell(recipe.TaskInference, []evaluation.EvalSummary{asked, alone, leaderboard}, descriptors)
	if cell.Contract.Protocol != raw || cell.Value != 0.19 || len(beside) != 2 {
		t.Fatalf("MMLU-Pro is headed by %q at %v with %d readings beside it; want the leaderboard reading, %s at 0.19, the others beside it", cell.Contract.Protocol, cell.Value, len(beside), raw)
	}
	if headed, _ := qualityCell(recipe.TaskInference, []evaluation.EvalSummary{asked, alone}, descriptors); headed.Contract.Protocol == alone.Prompting {
		t.Fatalf("with no leaderboard reading, the reading asked without demonstrations heads MMLU-Pro; want it never the headline")
	}
}
