package controllertrain

import (
	"strings"
	"testing"
	"unicode/utf8"

	"overgo/internal/scratchmodel"
	"overgo/internal/scratchmodeltest"
)

func TestControllerEvaluationInputs(t *testing.T) {
	spec := fixtureCorpusSpec()
	// Include every character and action in the small construction's vocabulary.
	for _, records := range [][]Record{spec.Train, spec.Holdout} {
		for i := range records {
			records[i].Prompt = "abcdefghijklmnopqrstuvwxyz 0123456789"
		}
	}
	corpus, err := Compile(spec)
	if err != nil {
		t.Fatal(err)
	}
	c, err := scratchmodel.Compile(scratchmodel.CorpusFacts{Documents: corpus.TrainingDocuments(), Seed: 17, Steps: 400}, scratchmodeltest.Profile(t))
	if err != nil {
		t.Fatal(err)
	}
	inputs, err := prepareEvaluationInputs(c, corpus)
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs.tokens) != len(spec.Holdout) || len(inputs.expected) != len(spec.Holdout) {
		t.Fatal("held-out denominator changed")
	}
	for i, candidates := range inputs.tokens {
		if inputs.expected[i] != spec.Holdout[i].Action || len(candidates) != len(corpus.Actions()) {
			t.Fatal("held-out action or candidate denominator changed")
		}
		for j, tokens := range candidates {
			document, err := corpus.CandidateDocument(spec.Holdout[i].Prompt, inputs.actions[j])
			if err != nil {
				t.Fatal(err)
			}
			if len(tokens) != utf8.RuneCountInString(document)+2 {
				t.Fatal("candidate sequence was truncated")
			}
		}
	}
	// A valid corpus may still exceed a particular construction's context.
	// Reject it during preparation, before a resident trainer is allocated.
	spec.Holdout[0].Prompt += " a"
	oversized, err := Compile(spec)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := prepareEvaluationInputs(c, oversized); err == nil || !strings.Contains(err.Error(), spec.Holdout[0].ID) {
		t.Fatalf("oversized evaluation lacked a named refusal: %v", err)
	}
}
