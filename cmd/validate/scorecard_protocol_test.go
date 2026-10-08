package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScorecardNamesTheAnswerProtocol holds what the printed scorecard says
// of how a multiple-choice reading was asked for. Each prompting protocol a
// text model's choice reading was scored under is named once, with the models
// read that way and the evaluation's own words for what the protocol does: a
// direct answer after an opened turn under the chat template, the likeliest
// option under raw completion. A reading that is not a choice, which has no
// chance to state, and another task's reading, whose pipeline shapes its own
// prompt, are named under neither; a scorecard with no choice reading says
// nothing of protocols.
func TestScorecardNamesTheAnswerProtocol(t *testing.T) {
	t.Parallel()
	chance := 0.25
	row := func(name string, task recipe.Task, protocol string, constant *float64) scorecard.Row {
		return scorecard.Row{Name: name, Task: task, Cells: map[metric.Name]scorecard.Cell{metric.Quality: {
			Value: 0.5, Label: "accuracy, suite (" + protocol + ")", Source: testutil.ArtifactID(t, artifact.KindEvaluation, name), Constant: constant,
			Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "suite", Protocol: protocol},
		}}}
	}
	chat, raw := evaluation.PromptingChatTemplate.Label(), evaluation.PromptingRawCompletion.Label()
	text := choiceProtocols([]scorecard.Row{
		row("an assistant", recipe.TaskInference, chat, &chance),
		row("another assistant", recipe.TaskInference, chat, &chance),
		row("a base model", recipe.TaskInference, raw, &chance),
		row("a model scored on passages", recipe.TaskInference, raw, nil),
		row("a model that reads pictures", recipe.TaskVQA, raw, &chance),
	})
	direct, known := evaluation.ChoiceProtocol(chat)
	likeliest, alsoKnown := evaluation.ChoiceProtocol(raw)
	if !known || !alsoKnown || !strings.Contains(direct, `"The answer is"`) || !strings.Contains(direct, "8 tokens") || !strings.Contains(direct, "thinking") {
		t.Fatalf("the chat-template protocol is described as %q, want its opener, its token bound and that it does not think", direct)
	}
	if _, unknown := evaluation.ChoiceProtocol("few-shot"); unknown {
		t.Fatal("a label that names no protocol was described")
	}
	for _, want := range []string{
		"- " + chat + " (an assistant, another assistant): " + direct + ".\n",
		"- " + raw + " (a base model): " + likeliest + ".\n",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("the protocols read:\n%s\nwant the line %q", text, want)
		}
	}
	if strings.Contains(text, "passages") || strings.Contains(text, "pictures") {
		t.Fatalf("a reading that is not a text model's choice is named among the protocols:\n%s", text)
	}
	if text := choiceProtocols([]scorecard.Row{row("a model scored on passages", recipe.TaskInference, raw, nil)}); text != "" {
		t.Fatalf("a scorecard with no choice reading says %q of protocols", text)
	}
}
