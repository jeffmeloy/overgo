package hfconvert

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
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

// keptSplit is a split as the rules the converter names make it: what the
// expression matches is kept as a piece of its own, and so is what lies between.
func keptSplit(regex string) map[string]any {
	return map[string]any{"type": "Split", "pattern": map[string]string{"Regex": regex}, "behavior": "Isolated", "invert": false}
}

// byteLevelStep is the last step of every declaration the converter names:
// bytes are mapped with no space added. useRegex says whether the step first
// splits by its own expression, which is the default rule's.
func byteLevelStep(useRegex bool) map[string]any {
	return map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": useRegex}
}

// declaration writes a pre-tokenizer as a tokenizer file holds it: several
// steps as a Sequence, one step as itself.
func declaration(t *testing.T, steps ...map[string]any) json.RawMessage {
	t.Helper()
	var value any = map[string]any{"type": "Sequence", "pretokenizers": steps}
	if len(steps) == 1 {
		value = steps[0]
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// TestConverterRefusesADeclarationItsRuleDoesNotCut holds the converter to the
// whole of what a tokenizer file declares, not to its split expressions alone.
// The rule a name selects at run time keeps what each split matches and what
// lies between, in the order declared, and then maps bytes with no space
// added. A declaration that removes or inverts a split, adds a space, splits
// again by the byte level's own expression, or carries a step the rule does
// not have is not known to be cut as the rule cuts: the model's own tokenizer
// may cut it otherwise, and that rule's name would then encode text to other
// IDs than the model was trained on. It must be refused. The same holds for a
// declaration with no split expression in it: only a byte level that splits
// by the default expression is the default rule. A step is held to its kind
// as well as to its options: a step of another kind that writes the options
// of a kept split or of the byte level is refused, and so are steps listed by
// a declaration that is not a Sequence.
func TestConverterRefusesADeclarationItsRuleDoesNotCut(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "fixtures", "pre_tokenizer", "deepseek-v3.json"))
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil {
		t.Fatal(err)
	}
	held := string(reference.PreTokenizer)
	if name, err := preTokenizerName(reference.PreTokenizer); err != nil || name != "deepseek-v3" {
		t.Fatalf("the held declaration is named %q, so a change to it proves nothing: %v", name, err)
	}
	unsaidInvert := keptSplit(qwen2SplitRegex)
	delete(unsaidInvert, "invert")
	// Each of these two writes every option of the step the rule has in its
	// place and is of another kind, so it is the kind alone that refuses it.
	otherLast := byteLevelStep(false)
	otherLast["type"] = "Metaspace"
	otherFirst := keptSplit(qwen2SplitRegex)
	otherFirst["type"] = "Punctuation"
	// These declare no split expression, or leave the shape every rule has:
	// kept splits, then one byte level.
	refusals := []struct{ what, declared string }{
		{"digits then a byte level that splits", `{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}`},
		{"a byte level alone that adds a space", `{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}`},
		{"a byte level alone that does not split", `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}`},
		{"a byte level that does not say whether it adds a space", `{"type":"ByteLevel","trim_offsets":true,"use_regex":true}`},
		{"a white space step alone", `{"type":"Whitespace"}`},
		{"a sequence of nothing", `{"type":"Sequence","pretokenizers":[]}`},
		{"a declaration of no type", `{}`},
		{"a known split with no byte level after it", string(declaration(t, keptSplit(qwen2SplitRegex)))},
		{"a byte level that is not the last step", string(declaration(t, byteLevelStep(true), byteLevelStep(false)))},
		{"a known split that does not say whether it inverts", string(declaration(t, unsaidInvert, byteLevelStep(false)))},
		{"a known split then a last step of another kind that adds no space", string(declaration(t, keptSplit(qwen2SplitRegex), otherLast))},
		{"a first step of another kind that keeps what a known expression matches", string(declaration(t, otherFirst, byteLevelStep(false)))},
	}
	// Each of these is the held declaration with one thing about it changed.
	for _, change := range []struct{ what, from, to string }{
		{"a first split that removes what it matches", `"behavior": "Isolated"`, `"behavior": "Removed"`},
		{"a first split that is inverted", `"invert": false`, `"invert": true`},
		{"a byte level that adds a space", `"add_prefix_space": false`, `"add_prefix_space": true`},
		{"a byte level that splits again by its own expression", `"use_regex": false`, `"use_regex": true`},
		{"a step the rule does not have", `"pretokenizers": [`, `"pretokenizers": [{"type":"Digits","individual_digits":true},`},
		{"a split on a string before the rule's own", `"pretokenizers": [`, `"pretokenizers": [{"type":"Split","pattern":{"String":" "},"behavior":"Isolated","invert":false},`},
		// Only a Sequence is known to run the steps it lists.
		{"the same steps listed by a declaration that is not a Sequence", `"type": "Sequence"`, `"type": "Whitespace"`},
	} {
		changed := strings.Replace(held, change.from, change.to, 1)
		if changed == held {
			t.Fatalf("the held declaration no longer writes %s, so %s was not tried", change.from, change.what)
		}
		refusals = append(refusals, struct{ what, declared string }{change.what, changed})
	}
	for _, refused := range refusals {
		if name, err := preTokenizerName(json.RawMessage(refused.declared)); err == nil {
			t.Errorf("%s was named %q", refused.what, name)
		}
	}
}

// TestConverterKeepsTheRuleOfADeclarationItKnows holds the refusals above from
// reaching a declaration the converter does know: each of these had its name
// before the converter began to refuse, and each keeps it. A file that
// declares nothing is cut by the default rule, as is a byte level that splits
// by the default expression, said or left to the default. The byte level's
// trim_offsets moves offsets and never IDs, so either value keeps the name;
// Qwen2.5-0.5B and Carbon-500M declare it false. The Qwen 3.5 shape is taken
// to be Qwen2's with its own expression: no Qwen 3.5 tokenizer file was at
// hand to read.
func TestConverterKeepsTheRuleOfADeclarationItKnows(t *testing.T) {
	t.Parallel()
	untrimmed := byteLevelStep(false)
	untrimmed["trim_offsets"] = false
	unsaidRegex := byteLevelStep(true)
	delete(unsaidRegex, "use_regex")
	for _, known := range []struct {
		what     string
		declared json.RawMessage
		want     string
	}{
		{"no declaration", nil, "gpt-2"},
		{"a null declaration", json.RawMessage("null"), "gpt-2"},
		{"a byte level alone that splits", declaration(t, byteLevelStep(true)), "gpt-2"},
		{"a byte level alone that leaves splitting to the default", declaration(t, unsaidRegex), "gpt-2"},
		{"the Qwen2 split", declaration(t, keptSplit(qwen2SplitRegex), untrimmed), "qwen2"},
		{"the Qwen 3.5 split", declaration(t, keptSplit(qwen35SplitRegex), untrimmed), "qwen35"},
		{"digit runs then the Llama 3 split", declaration(t, keptSplit(digitSplitRegex), keptSplit(llama3SplitRegex), byteLevelStep(false)), "llama-bpe"},
		{"the Kimi split", declaration(t, keptSplit(tokenizer.BPEPatternKimi), byteLevelStep(false)), "kimi"},
	} {
		if name, err := preTokenizerName(known.declared); err != nil || name != known.want {
			t.Errorf("%s is named %q, and its rule is %q: %v", known.what, name, known.want, err)
		}
	}
}
