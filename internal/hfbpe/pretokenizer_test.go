package hfbpe

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Literal declarations match the pinned external artifacts, independently of the
// production catalog's private constants. See PRETOKENIZER.md for provenance.
const (
	qwenSplitPattern       = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	threeDigitSplitPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

func TestDeclaredPipeline(t *testing.T) {
	byteLevel := func(prefix, regex bool) map[string]any {
		return map[string]any{"type": "ByteLevel", "add_prefix_space": prefix, "trim_offsets": true, "use_regex": regex}
	}
	split := func(pattern, behavior string, invert bool) map[string]any {
		return map[string]any{"type": "Split", "pattern": map[string]string{"Regex": pattern}, "behavior": behavior, "invert": invert}
	}
	sequence := func(nodes ...any) map[string]any { return map[string]any{"type": "Sequence", "pretokenizers": nodes} }
	for _, test := range []struct {
		name, text string
		pre        any
		want       []string
	}{
		{"gpt2", "Hello'S 1234!!\n next", byteLevel(false, true), []string{"Hello", "'", "S", " 1234", "!!", "\n", " next"}},
		{"qwen2", "Hello'S 1234!!\n next", sequence(split(qwenSplitPattern, "Isolated", false), byteLevel(false, false)), []string{"Hello", "'S", " ", "1", "2", "3", "4", "!!\n", " next"}},
		{"three-digit", "Hello'S 1234!!\n next", sequence(split(threeDigitSplitPattern, "Removed", true), byteLevel(false, false)), []string{"Hello", "'S", " ", "123", "4", "!!\n", " next"}},
		{"newline-lookahead", " \n  a", sequence(split(qwenSplitPattern, "Isolated", false), byteLevel(false, false)), []string{" \n", " ", " a"}},
		{"prefix", "hello", byteLevel(true, true), []string{" hello"}},
		{"existing-prefix", " hello", byteLevel(true, true), []string{" hello"}},
		{"tab-prefix", "\thello", byteLevel(true, true), []string{" ", "\t", "hello"}},
		{"no-regex", "Hi 123!", byteLevel(false, false), []string{"Hi 123!"}},
		{"default-regex", "Hi 123!", map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false}, []string{"Hi", " 123", "!"}},
		{"individual-digits", "Hey 123 friend!", sequence(map[string]any{"type": "Digits", "individual_digits": true}, byteLevel(false, false)), []string{"Hey ", "1", "2", "3", " friend!"}},
		{"numeric-run", "x１２Ⅷ½y", sequence(map[string]any{"type": "Digits", "individual_digits": false}, byteLevel(false, false)), []string{"x", "１２Ⅷ½", "y"}},
		{"prefix-each-segment", "a12b", sequence(map[string]any{"type": "Digits", "individual_digits": true}, byteLevel(true, false)), []string{" a", " 1", " 2", " b"}},
		{"nested", "a12b", sequence(sequence(map[string]any{"type": "Digits", "individual_digits": false}), sequence(byteLevel(false, false))), []string{"a", "12", "b"}},
		{"empty", "", byteLevel(true, true), nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			tok := loadTokenizerFixture(t, nil, test.pre, nil)
			if tok.preTokenizerErr != nil {
				t.Fatal(tok.preTokenizerErr)
			}
			got := tok.preTokenize(test.text)
			if !slices.Equal(got, test.want) {
				t.Fatalf("pieces=%q want=%q", got, test.want)
			}
			ids, err := tok.Encode(test.text)
			if err != nil || !slices.Equal(ids, byteIDs(strings.Join(test.want, ""))) {
				t.Fatalf("IDs=%v error=%v", ids, err)
			}
		})
	}
	for _, pre := range []any{
		[]string{"ByteLevel"},
		map[string]any{"type": "Whitespace"},
		sequence(),
		sequence(byteLevel(false, false), map[string]any{"type": "Digits", "individual_digits": true}),
		sequence(byteLevel(false, false), byteLevel(false, false)),
		map[string]any{"type": "Digits", "individual_digits": true},
		map[string]any{"type": "Digits"},
		map[string]any{"type": "ByteLevel"},
		map[string]any{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": nil},
		map[string]any{"type": "ByteLevel", "add_prefix_space": "false", "trim_offsets": false},
		sequence(split("unsupported", "Isolated", false), byteLevel(false, false)),
		sequence(split(qwenSplitPattern, "Removed", false), byteLevel(false, false)),
		sequence(split(qwenSplitPattern, "MergedWithNext", false), byteLevel(false, false)),
		sequence(map[string]any{"type": "Split", "pattern": map[string]string{"String": " "}, "behavior": "Isolated", "invert": false}, byteLevel(false, false)),
	} {
		tok := loadTokenizerFixture(t, nil, pre, nil)
		if _, err := tok.Encode("a"); err == nil {
			t.Fatalf("unsupported declaration accepted: %v", pre)
		}
		if got, err := tok.DecodeStrict(byteIDs("a")); err != nil || got != "a" {
			t.Fatalf("decode-only contract changed: %q %v", got, err)
		}
	}
	added := []map[string]any{{"id": 256, "content": "<raw>", "normalized": false, "special": false}}
	tok := loadTokenizerFixture(t, nil, byteLevel(true, false), added)
	want := append(byteIDs(" a"), 256)
	want = append(want, byteIDs(" b")...)
	if got, err := tok.Encode("a<raw>b"); err != nil || !slices.Equal(got, want) {
		t.Fatalf("raw token boundary: %v %v", got, err)
	}
	for _, flag := range []string{"single_word", "lstrip", "rstrip", "normalized"} {
		invalid := []map[string]any{{"id": 256, "content": "<raw>", "normalized": false, flag: true}}
		tok := loadTokenizerFixture(t, nil, byteLevel(false, false), invalid)
		if _, err := tok.Encode("a<raw>b"); err == nil {
			t.Fatalf("ignored added-token flag %s", flag)
		}
	}
	if _, err := tok.Encode("a\xff"); err == nil {
		t.Fatal("accepted invalid UTF-8 in declared pipeline")
	}
	t.Run("marker-conflict", func(t *testing.T) {
		directory := t.TempDir()
		data := `{"model":{"type":"BPE","vocab":{"a":0,"▁":1,"b":2},"merges":[]},"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}}`
		if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		marker, err := Load(directory)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := marker.Encode("a b"); err == nil {
			t.Fatal("declared byte pipeline silently used legacy marker encoding")
		}
		if got, err := marker.DecodeStrict([]int{0, 1, 2}); err != nil || got != "a b" {
			t.Fatalf("marker decoding changed: %q %v", got, err)
		}
	})
}

func TestDeclaredPipelineOracle(t *testing.T) {
	for _, source := range []struct{ name, vocabulary, expected string }{
		{"qwen2", "44c2f46b715f585c6ab513970e8a006bfa5badd6108560054921cf598d154d8c", "04c6278ac3bf07c4af8d80a28f7135fcb664bfc198d9e0845526579c7082d260"},
		{"gpt-2", "cedc56ca6e2e89f63e781696d1fd76b4b1d49e6720dee86463e915f6e90016ac", "619d4283967ae1cee640809b09c31228d919926024094e4b9b99d9727b38269c"},
	} {
		t.Run(source.name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("testdata", source.name+"-oracle.json"))
			if err != nil {
				t.Fatal(err)
			}
			var fixture struct {
				SourceCommit     string          `json:"source_commit"`
				VocabularySHA256 string          `json:"vocabulary_sha256"`
				InputSHA256      string          `json:"input_sha256"`
				ExpectedSHA256   string          `json:"expected_sha256"`
				Tokenizer        json.RawMessage `json:"tokenizer"`
				Cases            []struct {
					Text string `json:"text"`
					IDs  []int  `json:"ids"`
				} `json:"cases"`
			}
			if err := json.Unmarshal(raw, &fixture); err != nil {
				t.Fatal(err)
			}
			if fixture.SourceCommit != "42fc243060709331ff9b158a9ed2cbe37219ae83" || len(fixture.Cases) != 46 {
				t.Fatal("independent corpus provenance/cardinality differs")
			}
			if fixture.VocabularySHA256 != source.vocabulary || fixture.ExpectedSHA256 != source.expected || fixture.InputSHA256 != "a4f554d42f793610f44fd8e56c97356bc2692d427bf618228ff236059d6a8a19" {
				t.Fatal("independent corpus source identity differs")
			}
			directory := t.TempDir()
			if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), fixture.Tokenizer, 0600); err != nil {
				t.Fatal(err)
			}
			tok, err := Load(directory)
			if err != nil {
				t.Fatal(err)
			}
			for index, test := range fixture.Cases {
				got, err := tok.Encode(test.Text)
				if err != nil || !slices.Equal(got, test.IDs) {
					t.Fatalf("case=%d input=%q got=%v want=%v error=%v", index, test.Text, got, test.IDs, err)
				}
			}
		})
	}
}

func TestConsumerTokenIdentity(t *testing.T) {
	raw, err := os.ReadFile("testdata/consumer-identity.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures []struct {
		Consumer       string          `json:"consumer"`
		SourceSHA256   string          `json:"source_sha256"`
		BaselineCommit string          `json:"baseline_commit"`
		Tokenizer      json.RawMessage `json:"tokenizer"`
		Cases          []struct {
			Text string `json:"text"`
			IDs  []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	expected := map[string]struct {
		digest string
		cases  int
	}{
		"Krea":     {"be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506", 3},
		"Fractale": {"9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c", 3},
		"Granite":  {"ee87426e9a5ed085795bd4d634ff84de6ccdf9b463531f9c781451badcf8f55e", 2},
	}
	if len(fixtures) != len(expected) {
		t.Fatal("consumer census cardinality differs")
	}
	for _, fixture := range fixtures {
		t.Run(fixture.Consumer, func(t *testing.T) {
			identity, found := expected[fixture.Consumer]
			if !found || fixture.BaselineCommit != "3eac92e48aa6b06d73c7364b05616c34086803c9" || fixture.SourceSHA256 != identity.digest || len(fixture.Cases) != identity.cases {
				t.Fatal("consumer provenance/cardinality differs")
			}
			delete(expected, fixture.Consumer)
			directory := t.TempDir()
			if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), fixture.Tokenizer, 0600); err != nil {
				t.Fatal(err)
			}
			tok, err := Load(directory)
			if err != nil {
				t.Fatal(err)
			}
			for _, sample := range fixture.Cases {
				got, err := tok.Encode(sample.Text)
				if err != nil || !slices.Equal(got, sample.IDs) {
					t.Fatalf("retained input=%q IDs=%v want=%v error=%v", sample.Text, got, sample.IDs, err)
				}
			}
		})
	}
	t.Run("legacy", func(t *testing.T) {
		// A merge spanning space plus digits is forbidden by the old scanner
		// and allowed by declared GPT2 ByteLevel. This distinguishes both paths.
		directory := t.TempDir()
		vocab := `{"Ġ":0,"1":1,"2":2,"Ġ1":3,"Ġ12":4}`
		model := `{"type":"BPE","vocab":` + vocab + `,"merges":["Ġ 1","Ġ1 2"]}`
		for _, declaration := range []string{"", `,"pre_tokenizer":null`} {
			data := `{"model":` + model + declaration + `}`
			if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
			tok, err := Load(directory)
			if err != nil {
				t.Fatal(err)
			}
			if ids, err := tok.Encode(" 12"); err != nil || !slices.Equal(ids, []int{0, 1, 2}) {
				t.Fatalf("legacy Load changed: %v %v", ids, err)
			}
		}
		if err := os.WriteFile(filepath.Join(directory, "vocab.json"), []byte(vocab), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, "merges.txt"), []byte("#version: 0.2\nĠ 1\nĠ1 2\n"), 0600); err != nil {
			t.Fatal(err)
		}
		legacy, err := LoadSplit(directory)
		if err != nil {
			t.Fatal(err)
		}
		if ids, err := legacy.Encode(" 12"); err != nil || !slices.Equal(ids, []int{0, 1, 2}) {
			t.Fatalf("legacy LoadSplit changed: %v %v", ids, err)
		}
		data := `{"model":` + model + `,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true}}`
		if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
		declared, err := Load(directory)
		if err != nil {
			t.Fatal(err)
		}
		if ids, err := declared.Encode(" 12"); err != nil || !slices.Equal(ids, []int{4}) {
			t.Fatalf("declared ByteLevel ignored: %v %v", ids, err)
		}
	})
}
