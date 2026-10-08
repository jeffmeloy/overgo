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
	}, nil)
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
	if text := choiceProtocols([]scorecard.Row{row("a model scored on passages", recipe.TaskInference, raw, nil)}, nil); text != "" {
		t.Fatalf("a scorecard with no choice reading says %q of protocols", text)
	}
}

// TestScorecardSetsASampleAskedAnotherWayBesideItsReading holds what the
// printed scorecard says where some of a reading's questions were also asked
// another way. The sample stands under the protocols, with what each way
// answers of the same questions, how many the two differ on and how many of
// those only the other way gets right, the figure the model's maker publishes
// where there is one, how it was asked, what a question and its answer ran to,
// how many answers ran to the limit, and what produced it; and every count is
// counted from the outcomes, here five questions of which the reading's
// protocol answers two and the other way four, one at the limit. A sample is
// not shown for a model the scorecard does not hold, for one whose reading was
// scored under another protocol than the sample names, or on a scorecard that
// shows no choice reading.
//
// The sample the repository holds is read too: it stands beside a reading the
// evaluation describes, and no answer in it ran to the limit, so its count is
// not of answers cut short.
func TestScorecardSetsASampleAskedAnotherWayBesideItsReading(t *testing.T) {
	t.Parallel()
	chance := 0.25
	row := func(name string, protocol string, constant *float64) scorecard.Row {
		return scorecard.Row{Name: name, Task: recipe.TaskInference, Cells: map[metric.Name]scorecard.Cell{metric.Quality: {
			Value: 0.5, Label: "accuracy, suite (" + protocol + ")", Source: testutil.ArtifactID(t, artifact.KindEvaluation, name), Constant: constant,
			Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "suite", Protocol: protocol},
		}}}
	}
	chat, raw := evaluation.PromptingChatTemplate.Label(), evaluation.PromptingRawCompletion.Label()
	sample := func(model, protocol string, published float64) scorecard.ProtocolSample {
		made := scorecard.ProtocolSample{
			Model: model, Task: recipe.TaskInference, Protocol: protocol, Way: "left to think", Asked: "thinking on", Limit: 500, ProducedBy: "a run",
			Cases: []scorecard.SampledCase{
				{Name: "both", Answer: 1, Reading: 1, Sample: 1, Tokens: 100},
				{Name: "the reading only", Answer: 2, Reading: 2, Sample: 0, Tokens: 200},
				{Name: "the other way only", Answer: 3, Reading: 0, Sample: 3, Tokens: 300},
				{Name: "the other way, the reading unanswered", Answer: 0, Reading: -1, Sample: 0, Tokens: 400},
				{Name: "the other way again", Answer: 2, Reading: 1, Sample: 2, Tokens: 500},
			},
		}
		if published != 0 {
			made.Published, made.PublishedBy = published, "its card"
		}
		return made
	}
	rows := []scorecard.Row{row("an assistant", chat, &chance), row("a base model", raw, &chance)}
	text := choiceProtocols(rows, []scorecard.ProtocolSample{
		sample("an assistant", chat, 0.8),
		sample("a base model", chat, 0),
		sample("a model not scored", chat, 0),
	})
	want := "- an assistant left to think, on 5 of its questions: 4 right (0.8 ±0.18), where the " + chat + " protocol answers 2 of the same 5 (0.4 ±0.22); " +
		"the two differ on 4 questions, 3 of them right only this way. Its maker publishes 0.8 for the suite (its card). " +
		"Asked: thinking on. A question and its answer ran to 300 tokens on average, and 1 of the 5 ran to the limit of 500 tokens. From a run.\n"
	if !strings.Contains(text, want) {
		t.Fatalf("the protocols read:\n%s\nwant the line\n%s", text, want)
	}
	if strings.Contains(text, "a base model left to think") || strings.Contains(text, "a model not scored") {
		t.Fatalf("a sample is shown for a reading under another protocol, or for a model the scorecard does not hold:\n%s", text)
	}
	if text := choiceProtocols(rows, []scorecard.ProtocolSample{sample("an assistant", chat, 0)}); strings.Contains(text, "Its maker publishes") {
		t.Fatalf("a sample with no published figure names one:\n%s", text)
	}
	if text := choiceProtocols([]scorecard.Row{row("an assistant", chat, nil)}, []scorecard.ProtocolSample{sample("an assistant", chat, 0)}); text != "" {
		t.Fatalf("a scorecard with no choice reading says %q of a sample", text)
	}

	held, err := scorecard.ProtocolSamples()
	if err != nil {
		t.Fatal(err)
	}
	if len(held) == 0 {
		t.Fatal("the repository holds no sample asked another way")
	}
	for _, sample := range held {
		if _, known := evaluation.ChoiceProtocol(sample.Protocol); !known {
			t.Errorf("the sample for %s stands beside a reading under %q, which the evaluation does not describe", sample.Model, sample.Protocol)
		}
		tally := sample.Tally()
		if tally.Sample+tally.ReadingOnly > tally.Cases || tally.Reading+tally.SampleOnly > tally.Cases || tally.MeanTokens <= 0 || tally.Cut != 0 {
			t.Errorf("the sample for %s counts to %+v", sample.Model, tally)
		}
		t.Logf("%s %s: %d of %d against %d by the %s protocol; %d and %d only one way; %d tokens on average",
			sample.Model, sample.Way, tally.Sample, tally.Cases, tally.Reading, sample.Protocol, tally.SampleOnly, tally.ReadingOnly, tally.MeanTokens)
	}
}
