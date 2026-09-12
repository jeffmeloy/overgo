package evaluation

import (
	_ "embed"
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
)

//go:embed testdata/ifeval_letter_oracle.json
var ifevalLetterOracle []byte

func TestIFEvalNativeParameterAcceptance(t *testing.T) {
	t.Run("legacy omitted operands retain identity", func(t *testing.T) {
		legacy := json.RawMessage(`{"kind":"instruction-rules","schema":"legacy","source":"fixture","cases":[{"name":"one","prompt":"Count cats.","max_tokens":1,"rules":[{"name":"cats","kind":"substring-count","values":["cat"],"count":{"relation":"at-least","value":2}}]}]}`)
		var suite InstructionRulesSuite
		if err := json.Unmarshal(legacy, &suite); err != nil {
			t.Fatal(err)
		}
		compiled, err := CompileInstructionRules(suite)
		if err != nil {
			t.Fatal(err)
		}
		identity, err := artifact.JSONID(artifact.KindProfile, legacy)
		if err != nil || identity != compiled.identity {
			t.Fatal("omitted loose operands changed legacy identity")
		}
	})
	var oracle struct {
		ReferenceVersion string `json:"reference_version"`
		Sources          map[string]string
		Cases            []struct {
			Kwargs        map[string]json.RawMessage
			Response      string
			Strict, Loose []bool
		}
	}
	if err := json.Unmarshal(ifevalLetterOracle, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.ReferenceVersion != "0.4.9.1" || len(oracle.Cases) != 96 || oracle.Sources["instructions.py"] != "1556285b56cb81a7a1ada79327cd8797681aed004160c7f87b3a94f3e10bae32" || oracle.Sources["utils.py"] != "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632" || oracle.Sources["instructions_util.py"] != "e8c4d9187bac1482d93941fb46469609c3ae78d896195bfcb300a0707d492567" {
		t.Fatal("native letter source or denominator changed")
	}
	for index, c := range oracle.Cases {
		rules, ok := ifevalRuleFor("keywords:letter_frequency", c.Kwargs)
		if !ok || len(rules) != 1 {
			t.Fatalf("case %d not mapped", index)
		}
		rule, err := compileInstructionRule(rules[0])
		if err != nil {
			t.Fatal(err)
		}
		strict, loose, err := evaluateInstructionViews(c.Response, []compiledInstructionRule{rule})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(strict, c.Strict) || !slices.Equal(loose, c.Loose) {
			t.Errorf("case %d %q: strict=%v/%v loose=%v/%v", index, c.Response, strict, c.Strict, loose, c.Loose)
		}
	}
	t.Run("views and owned criteria", func(t *testing.T) {
		makeRule := func(letter string) InstructionRule {
			return InstructionRule{Name: "letter", Kind: ifevalLetters, Values: []string{letter}, Count: &CountRule{Relation: RelationAtLeast, Value: 2}}
		}
		strict, loose := makeRule("a"), makeRule("b")
		strict.Loose = &loose
		suite := InstructionRulesSuite{Kind: InstructionRulesKind, Schema: "resolved-views", Source: "fixture", Cases: []InstructionRulesCase{{Name: "one", Prompt: "Count letters.", MaxTokens: 1, Rules: []InstructionRule{strict}}}}
		compiled, err := CompileInstructionRules(suite)
		if err != nil {
			t.Fatal(err)
		}
		loose.Values[0] = "c"
		loose.Count.Value = 4
		gotStrict, gotLoose, err := evaluateInstructionViews("aa", compiled.rules[0])
		if err != nil {
			t.Fatal(err)
		}
		if !gotStrict[0] || gotLoose[0] {
			t.Fatal("loose operands did not remain independent")
		}
		gotStrict, gotLoose, err = evaluateInstructionViews("bb", compiled.rules[0])
		if err != nil {
			t.Fatal(err)
		}
		if gotStrict[0] || !gotLoose[0] {
			t.Fatal("caller mutation changed resolved loose operands")
		}
		identity, err := artifact.JSONID(artifact.KindProfile, compiled.suite)
		if err != nil || identity != compiled.identity {
			t.Fatal("compiled parameter identity drifted")
		}
		for _, mutation := range []string{"name", "kind", "nested", "invalid"} {
			t.Run(mutation, func(t *testing.T) {
				a, b := makeRule("a"), makeRule("b")
				a.Loose = &b
				switch mutation {
				case "name":
					b.Name = "other"
				case "kind":
					b.Kind = RuleSuffix
				case "nested":
					c := makeRule("c")
					b.Loose = &c
				case "invalid":
					b.Values[0] = "#"
				}
				if _, err := compileInstructionRule(a); err == nil {
					t.Fatal("invalid view accepted")
				}
			})
		}
	})
}
