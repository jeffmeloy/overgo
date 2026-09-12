package evaluation

import (
	_ "embed"
	"encoding/json"
	"slices"
	"testing"
	"unicode"

	"overgo/internal/artifact"
)

//go:embed testdata/ifeval_paragraph_oracle.json
var ifevalParagraphOracle []byte

func TestIFEvalIndexedParagraphAcceptance(t *testing.T) {
	var oracle struct {
		ReferenceVersion string `json:"reference_version"`
		Sources          map[string]string
		Cases            []struct {
			Kwargs        map[string]json.RawMessage
			Response      string
			Strict, Loose []bool
		}
	}
	if err := json.Unmarshal(ifevalParagraphOracle, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.ReferenceVersion != "0.4.9.1" || len(oracle.Cases) != 41 || oracle.Sources["instructions.py"] != "1556285b56cb81a7a1ada79327cd8797681aed004160c7f87b3a94f3e10bae32" || oracle.Sources["instructions_util.py"] != "e8c4d9187bac1482d93941fb46469609c3ae78d896195bfcb300a0707d492567" || oracle.Sources["utils.py"] != "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632" {
		t.Fatal("native source or denominator changed")
	}
	const id = "length_constraints:nth_paragraph_first_word"
	compared, refused := 0, 0
	for index, c := range oracle.Cases {
		var word string
		if err := json.Unmarshal(c.Kwargs["first_word"], &word); err != nil {
			t.Fatal(err)
		}
		unsupported := false
		for _, r := range word {
			unsupported = unsupported || r > unicode.MaxASCII
		}
		rules, mapped := ifevalRuleFor(id, c.Kwargs)
		if unsupported {
			if mapped {
				t.Fatalf("%d: unsupported vocabulary admitted", index)
			}
			refused++
			continue
		}
		if !mapped || len(rules) != 1 {
			t.Fatalf("%d: instruction missing or duplicated", index)
		}
		assembled, dropped, err := assembleIFEvalSuite([]storeCase{{entry: "ifeval/default/train", subset: "default", ordinal: index, fields: rawFields(t, map[string]any{"prompt": "Follow the paragraph instruction.", "instruction_id_list": []string{id}, "kwargs": []map[string]json.RawMessage{c.Kwargs}})}})
		if err != nil || dropped != 0 {
			t.Fatalf("%d: dropped=%d err=%v", index, dropped, err)
		}
		compiled, err := CompileInstructionRules(assembled.(InstructionRulesSuite))
		if err != nil {
			t.Fatal(err)
		}
		if len(compiled.rules) != 1 || len(compiled.rules[0]) != 1 {
			t.Fatal("paragraph denominator expanded")
		}
		strict, loose := evaluateInstructionViews(c.Response, compiled.rules[0])
		if !slices.Equal(strict, c.Strict) || !slices.Equal(loose, c.Loose) {
			t.Errorf("%d %q: strict=%v/%v loose=%v/%v", index, c.Response, strict, c.Strict, loose, c.Loose)
		}
		compared++
	}
	if compared != 37 || refused != 4 {
		t.Fatalf("compared=%d refused=%d", compared, refused)
	}
	for _, operands := range []map[string]any{
		{}, {"num_paragraphs": 2, "nth_paragraph": 0, "first_word": "cat"},
		{"num_paragraphs": 2, "nth_paragraph": -1, "first_word": "cat"},
		{"num_paragraphs": 2, "nth_paragraph": 3, "first_word": "cat"},
		{"num_paragraphs": 2, "nth_paragraph": 1.5, "first_word": "cat"},
		{"num_paragraphs": 0, "nth_paragraph": 1, "first_word": "cat"},
		{"num_paragraphs": -1, "nth_paragraph": 1, "first_word": "cat"},
		{"num_paragraphs": 2, "nth_paragraph": 1, "first_word": ""},
		{"num_paragraphs": 2, "nth_paragraph": 1, "first_word": nil},
	} {
		if _, mapped := ifevalRuleFor(id, rawFields(t, operands)); mapped {
			t.Errorf("unresolved operands accepted: %v", operands)
		}
	}
	t.Run("ordinal is owned and part of acceptance identity", func(t *testing.T) {
		rule := InstructionRule{Name: id, Kind: ifevalIndexedParagraph, Values: []string{"cat"}, Count: &CountRule{Relation: RelationEqual, Value: 2}, Ordinal: 2}
		suite := InstructionRulesSuite{Kind: InstructionRulesKind, Schema: "paragraph-ownership", Source: "fixture", Cases: []InstructionRulesCase{{Name: "paragraph", Prompt: "Write two paragraphs.", MaxTokens: 1, Rules: []InstructionRule{rule}}}}
		compiled, err := CompileInstructionRules(suite)
		if err != nil {
			t.Fatal(err)
		}
		suite.Cases[0].Rules[0].Ordinal = 1
		other, err := CompileInstructionRules(suite)
		if err != nil || compiled.identity == other.identity {
			t.Fatal("ordinal missing from acceptance identity")
		}
		suite.Cases[0].Rules[0].Count.Value = 1
		suite.Cases[0].Rules[0].Values[0] = "dog"
		strict, _ := evaluateInstructionViews("one\n\ncat meows", compiled.rules[0])
		if !slices.Equal(strict, []bool{true}) {
			t.Fatal("caller mutation changed paragraph acceptance")
		}
		identity, err := artifact.JSONID(artifact.KindProfile, compiled.suite)
		if err != nil || identity != compiled.identity {
			t.Fatal("published criteria changed")
		}
	})
	t.Run("legacy rule bytes stay unchanged", func(t *testing.T) {
		rule := InstructionRule{Name: "comma", Kind: RuleNoComma}
		data, err := json.Marshal(rule)
		if err != nil || string(data) != `{"name":"comma","kind":"no-comma"}` {
			t.Fatalf("legacy bytes changed: %s %v", data, err)
		}
		rule.Ordinal = 1
		if _, err := compileInstructionRule(rule); err == nil {
			t.Fatal("unused ordinal silently accepted")
		}
	})
}
