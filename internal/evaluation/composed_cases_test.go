package evaluation

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestImageSuitesHoldCasesOfThreeToTenAtoms holds the picture suites off their ceiling: on 2026-11-08 SenseNova read
// 100 of 100 drawn texts and Krea-2-Turbo 0.94 of TIFA's questions. Each suite composes cases of 3 to 10 atoms from its
// own atoms, one question an atom whose answer is among its choices, every count from 3 to 10 among the sixteen
// composed by the suites' seed; the suite of them compiles as a suite of choices; and the cases are frozen by their
// digest, so a change to how they are composed is a change to the measure, made by hand here.
func TestImageSuitesHoldCasesOfThreeToTenAtoms(t *testing.T) {
	t.Parallel()
	for _, suite := range []struct {
		name    string
		compose func(int, uint64) ([]TIFAItem, error)
		build   func([]TIFAItem, map[string]string, string) (GeneratedAnswerSuite, error)
		digest  string
	}{
		{"TIFA", ComposeTIFA, BuildTIFAQuestions, "8d320fffd136dc0a2d38cf18559dee23a02da7375d085107cd553c7bd72b66ed"},
		{"drawn text", ComposeDrawnText, BuildDrawnText, "9a539d291e35abf6b413b58bbad44e5fb1fe744de5fff9bad2a0ce482394ff79"},
	} {
		items := testcheck.Must(suite.compose(16, 17))(t)
		images, counts := map[string]string{}, map[int]bool{}
		for _, item := range items {
			counts[len(item.Questions)] = true
			for _, question := range item.Questions {
				if !slices.Contains(question.Choices, question.Answer) {
					t.Fatalf("%s's composed %s asks %q of %v, whose answer %q is no choice", suite.name, item.ID, question.Question, question.Choices, question.Answer)
				}
			}
			images[item.ID] = strings.Repeat("c", 64) + imageExtension
		}
		for atoms := fewestAtoms; atoms <= mostAtoms; atoms++ {
			if !counts[atoms] || len(counts) != mostAtoms-fewestAtoms+1 {
				t.Fatalf("%s's composed cases hold %v atoms, want every count from %d to %d and no other", suite.name, counts, fewestAtoms, mostAtoms)
			}
		}
		built := testcheck.Must(suite.build(items, images, "judge-a"))(t)
		if compiled, err := CompileGeneratedAnswer(built); err != nil || !compiled.suite.choices() {
			t.Fatalf("%s's composed suite does not compile as a suite of choices: %v", suite.name, err)
		}
		digest := sha256.Sum256(testcheck.Must(json.Marshal(items))(t))
		if got := hex.EncodeToString(digest[:]); got != suite.digest {
			t.Errorf("%s's composed cases are %s, not the frozen %s: a change to how they are composed changes the measure", suite.name, got, suite.digest)
		}
	}
}
