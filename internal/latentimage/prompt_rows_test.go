package latentimage

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/hfbpe"
	"overgo/internal/testcheck"
)

// TestPromptWeightsBindTheRowsTheyExecute holds a weighted prompt's rows to the tokens the encoder runs (prompt-weights
// /rows, SQA-382), on Krea-2-Turbo's own tokenizer and template: a prompt was tokenized a weighted stretch at a time, so
// markup at weight 1 split what the plain prompt merges (" red" into " " and "red"), and its weights were placed on as
// many rows as it had tokens, so a prompt long enough to reach the template's suffix scaled the suffix's rows. A prompt
// whose markup weighs 1 renders the plain prompt's rows, every one weighing 1; a prompt whose markup weighs 2 renders
// them too, a weighted word's rows at 2; and a weighted prompt longer than its rows leaves the suffix's rows at 1.
func TestPromptWeightsBindTheRowsTheyExecute(t *testing.T) {
	t.Parallel()
	tok := testcheck.Must(hfbpe.Load(filepath.Join(kreaDirOrSkip(t), "tokenizer")))(t)
	template := kreaProfileOrSkip(t).Prompt
	render := func(prompt string) textInput {
		t.Helper()
		return testcheck.Must(renderTextInput(tok, prompt, template))(t)
	}
	plain := render("a red fox in the snow")
	ones := render("a (red:1) fox in the snow")
	if !slices.Equal(ones.IDs, plain.IDs) || slices.ContainsFunc(ones.Weights, func(weight float32) bool { return weight != unweighted }) {
		t.Errorf("a prompt whose markup weighs 1 rendered %d rows unlike the plain prompt's, or weighed %v", len(ones.IDs), ones.Weights)
	}
	twos := render("a (red:2) fox in the snow")
	if !slices.Equal(twos.IDs, plain.IDs) || !slices.Contains(twos.Weights, 2) {
		t.Errorf("a prompt that weighs red at 2 rendered rows unlike the plain prompt's, or weighed no row at 2: %v", twos.Weights)
	}

	long := render(strings.Repeat("(fox:2) ", template.MaxTokens))
	if suffix := long.Weights[len(long.Weights)-template.SuffixTokens:]; slices.ContainsFunc(suffix, func(weight float32) bool { return weight != unweighted }) {
		t.Errorf("a weighted prompt longer than its rows weighed the template's suffix rows %v; want each at 1", suffix)
	}
}
