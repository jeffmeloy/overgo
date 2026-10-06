package evaluation

import "testing"

// TestLeadingLetterReadsTheLetterAnAnswerOpensWith holds the transform to the
// answers it reads as one letter, and leaves alone those that open with a word.
func TestLeadingLetterReadsTheLetterAnAnswerOpensWith(t *testing.T) {
	t.Parallel()
	for answer, want := range map[string]string{
		"B":                 "B",
		" B.":               "B",
		"B) the second":     "B",
		"C because it fits": "C",
		"The first":         "The first",
		"":                  "",
		"7":                 "7",
	} {
		if got := normalizeAnswer(answer, []string{transformLeadingLetter}); got != want {
			t.Errorf("%q read as %q, want %q", answer, got, want)
		}
	}
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: "fixture/v1", Source: "fixture", Transforms: []string{TransformTrimSpace, transformLeadingLetter, transformLowercase},
		Cases: []GeneratedAnswerCase{{Name: "case", Prompt: "prompt", MaxTokens: 1, Answers: []string{"B"}}},
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil {
		t.Fatalf("a suite that reads answers by their leading letter does not compile: %v", err)
	}
}
