package evaluation

import (
	"bytes"
	"errors"
	"slices"
	"testing"

	"github.com/dlclark/regexp2/v2"
)

// TestIFEvalLanguageRefusals holds the compiled language checker to refusing
// a changed profile, an unresolved language and an execution error dressed
// as a native outcome. It reads no retained evidence, so the landing runs it;
// the parity against the native oracle is the cycle's.
func TestIFEvalLanguageRefusals(t *testing.T) {
	t.Parallel()
	p, err := compiledIFEvalLanguage()
	if err != nil {
		t.Fatal(err)
	}
	t.Run("changed profiles and unresolved languages are refused", func(t *testing.T) {
		if _, err := loadIFEvalLanguage(bytes.Replace(ifevalLanguageProfile, []byte("1.0.9"), []byte("1.0.8"), 1)); err == nil {
			t.Fatal("changed language profile accepted")
		}
		for _, values := range [][]string{nil, {""}, {"not-a-language"}, {"en", "fr"}} {
			if _, err := compileInstructionRule(instructionRule{Name: "language", Kind: ifevalLanguageRule, Values: values}); err == nil {
				t.Fatal("unresolved language accepted")
			}
		}
	})
	t.Run("execution errors are not native no-feature outcomes", func(t *testing.T) {
		broken := *p
		broken.url, err = regexp2.Compile(`(?:^){100}`, regexp2.OptionMaxBacktrackingStackSize(32))
		if err != nil {
			t.Fatal(err)
		}
		rule, err := compileInstructionRule(instructionRule{Name: "language", Kind: ifevalLanguageRule, Values: []string{"en"}})
		if err != nil {
			t.Fatal(err)
		}
		rule.language = &broken
		strict, loose, err := evaluateInstructionViews("HELLO", []compiledInstructionRule{rule})
		if !errors.Is(err, regexp2.ErrBacktrackingStackLimit) || strict != nil || loose != nil {
			t.Fatal("execution error became a native verdict")
		}
		rule.language = p
		strict, loose, err = evaluateInstructionViews("This is a complete English sentence.", []compiledInstructionRule{rule})
		if err != nil || !slices.Equal(strict, []bool{true}) || !slices.Equal(loose, []bool{true}) {
			t.Fatal("language evaluation failed to recover")
		}
	})
}
