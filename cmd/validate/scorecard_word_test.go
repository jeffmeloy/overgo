package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScorecardSaysHowEachWordErrorRateWasCounted holds the printed scorecard
// to naming, for each way a word error rate on it was counted, the readings
// counted that way: two recognizers and a speech model counted with the
// recognizers' own normalizer are named together, and a projector counted
// over the words as written is named apart, with what that way charges. A
// reading that is no word error rate, and a word error rate nothing measured,
// are named nowhere, and a scorecard with no such rate says nothing.
func TestScorecardSaysHowEachWordErrorRateWasCounted(t *testing.T) {
	t.Parallel()
	row := func(name string, task recipe.Task, rate, protocol string, measured bool) scorecard.Row {
		cell := scorecard.Cell{Reason: "nothing measures it", Contract: scorecard.Contract{Metric: rate, Protocol: protocol}}
		if measured {
			cell = scorecard.Cell{Value: 0.02, Label: rate, Source: testutil.ArtifactID(t, artifact.KindEvaluation, name), Contract: scorecard.Contract{Metric: rate, Protocol: protocol}}
		}
		return scorecard.Row{Name: name, Task: task, Cells: map[metric.Name]scorecard.Cell{metric.Quality: cell}}
	}
	text := wordCounting([]scorecard.Row{
		row("a recognizer", recipe.TaskTranscription, "wer", "raw-completion, whisper-english", true),
		row("a projector", recipe.TaskProjection, "wer", "chat-template, lowercase, strip-punctuation, collapse-whitespace", true),
		row("a speaker", recipe.TaskSpeech, "word_error_rate", "raw-completion, whisper-english", true),
		row("a speaker counted plainly", recipe.TaskSpeech, "word_error_rate", "raw-completion", true),
		row("an assistant", recipe.TaskInference, "share", "chat-template", true),
		row("a recognizer nothing scored", recipe.TaskTranscription, "wer", "raw-completion, whisper-english", false),
	})
	lines := strings.Split(strings.TrimSpace(text), "\n")
	if len(lines) != 3 || lines[0] != "How the word error rates were counted:" {
		t.Fatalf("the counting reads:\n%s", text)
	}
	oneWay, plain := lines[1], lines[2]
	if !strings.HasPrefix(oneWay, "- a recognizer transcription, a speaker speech: ") || !strings.Contains(oneWay, "both written by the English normalizer") {
		t.Errorf("the readings counted the recognizers' way read: %s", oneWay)
	}
	if !strings.HasPrefix(plain, "- a projector projection, a speaker counted plainly speech: ") || !strings.Contains(plain, `"10" for "ten"`) || !strings.Contains(plain, "not to be read against") {
		t.Errorf("the readings counted as written read: %s", plain)
	}
	for _, absent := range []string{"an assistant", "nothing scored"} {
		if strings.Contains(text, absent) {
			t.Errorf("%q is named among the word error rates:\n%s", absent, text)
		}
	}
	if text := wordCounting([]scorecard.Row{row("an assistant", recipe.TaskInference, "share", "chat-template", true)}); text != "" {
		t.Fatalf("a scorecard with no word error rate says %q of counting them", text)
	}
}
