package hfconvert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestConverterNamesASplittingRuleByTheFileThatHoldsIt holds the name the
// converter writes for a tokenizer's splitting rule to the tokenizer file the
// runtime's rule of that name is held to. Each fixture of
// fixtures/pre_tokenizer is named for a rule and carries, beside the pieces
// internal/tokenizer holds that rule to, the pre-tokenizer of the model file
// they were cut by; the converter must give that pre-tokenizer the fixture's
// name. A file converted under another rule's name is cut by another rule,
// which is what the first conversion of Unlimited-OCR was. A sequence of
// splits the converter does not know is refused, not guessed at.
func TestConverterNamesASplittingRuleByTheFileThatHoldsIt(t *testing.T) {
	t.Parallel()
	fixtures, err := filepath.Glob(filepath.Join(testutil.RepoRoot(t), "fixtures", "pre_tokenizer", "*.json"))
	if err != nil || len(fixtures) == 0 {
		t.Fatalf("no rule's tokenizer is on file: %v", err)
	}
	for _, fixture := range fixtures {
		raw, err := os.ReadFile(fixture)
		if err != nil {
			t.Fatal(err)
		}
		var reference struct {
			Rule         string          `json:"rule"`
			PreTokenizer json.RawMessage `json:"pre_tokenizer"`
		}
		if err := json.Unmarshal(raw, &reference); err != nil || len(reference.PreTokenizer) == 0 {
			t.Fatalf("%s holds no pre-tokenizer: %v", fixture, err)
		}
		name, err := preTokenizerName(reference.PreTokenizer)
		if err != nil || name != reference.Rule {
			t.Errorf("the converter names the pre-tokenizer of %s %q, and the rule held to it is %q: %v", filepath.Base(fixture), name, reference.Rule, err)
		}
		// One character more in a split is another rule, and has no name.
		if name, err := preTokenizerName(json.RawMessage(strings.Replace(string(reference.PreTokenizer), "{1,3}", "{1,4}", 1))); err == nil {
			t.Errorf("a sequence of splits the converter does not know was named %q", name)
		}
	}
}
