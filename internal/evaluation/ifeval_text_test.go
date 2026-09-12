package evaluation

import (
	_ "embed"
	"encoding/json"
	"slices"
	"testing"
)

//go:embed testdata/ifeval_text_oracle.json
var ifevalTextOracle []byte

func TestIFEvalTextContractAcceptance(t *testing.T) {
	var oracle struct {
		ReferenceVersion string `json:"reference_version"`
		Sources          map[string]string
		Cases            []struct {
			Instruction   string
			Kwargs        map[string]json.RawMessage
			Response      string
			Strict, Loose []bool
		}
	}
	if err := json.Unmarshal(ifevalTextOracle, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.ReferenceVersion != "0.4.9.1" || len(oracle.Cases) != 25 || oracle.Sources["instructions.py"] != "1556285b56cb81a7a1ada79327cd8797681aed004160c7f87b3a94f3e10bae32" || oracle.Sources["utils.py"] != "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632" {
		t.Fatal("native checker or denominator changed")
	}
	legacyKinds := map[string]string{"keywords:existence": RuleContainsAll, "keywords:forbidden_words": RuleExcludesAll, "startend:end_checker": RuleSuffix}
	corrected := 0
	for index, c := range oracle.Cases {
		assembled, dropped, err := assembleIFEvalSuite([]storeCase{{entry: "ifeval/default/train", subset: "default", ordinal: index, fields: rawFields(t, map[string]any{"prompt": "Follow the stated instruction.", "instruction_id_list": []string{c.Instruction}, "kwargs": []map[string]json.RawMessage{c.Kwargs}})}})
		if err != nil || dropped != 0 {
			t.Fatalf("assemble: dropped=%d error=%v", dropped, err)
		}
		suite := assembled.(InstructionRulesSuite)
		compiled, err := CompileInstructionRules(suite)
		if err != nil {
			t.Fatal(err)
		}
		if len(compiled.rules) != 1 || len(compiled.rules[0]) != 1 {
			t.Fatal("one instruction must produce one verdict")
		}
		strict, loose := evaluateInstructionViews(c.Response, compiled.rules[0])
		if !slices.Equal(strict, c.Strict) || !slices.Equal(loose, c.Loose) {
			t.Errorf("%d %s %q strict=%v/%v loose=%v/%v", index, c.Instruction, c.Response, strict, c.Strict, loose, c.Loose)
		}
		legacy := suite
		legacy.Cases = slices.Clone(suite.Cases)
		legacy.Cases[0].Rules = slices.Clone(suite.Cases[0].Rules)
		legacy.Cases[0].Rules[0].Kind = legacyKinds[c.Instruction]
		prior, err := CompileInstructionRules(legacy)
		if err != nil {
			t.Fatal(err)
		}
		if prior.identity == compiled.identity || prior.dataset == compiled.dataset || prior.split == compiled.split {
			t.Fatal("scorer correction reused prior identity")
		}
		oldStrict, oldLoose := evaluateInstructionViews(c.Response, prior.rules[0])
		if !slices.Equal(oldStrict, c.Strict) || !slices.Equal(oldLoose, c.Loose) {
			corrected++
		}
	}
	if corrected != 14 {
		t.Fatalf("native counterexample denominator=%d want 14", corrected)
	}
	for _, c := range []InstructionRule{
		{Name: "unsupported", Kind: ifevalKeywordsRule, Values: []string{"a[bc]"}},
		{Name: "unsupported", Kind: ifevalKeywordsRule, Values: []string{"a|b"}},
		{Name: "unsupported", Kind: ifevalKeywordsRule, Values: []string{`cat\b`}},
		{Name: "unsupported", Kind: ifevalKeywordsRule, Values: []string{"café"}},
		{Name: "unsupported", Kind: ifevalForbiddenRule, Values: []string{".cat"}},
		{Name: "unsupported", Kind: ifevalEndRule, Values: []string{"Τέλος"}},
		{Name: "empty", Kind: ifevalKeywordsRule},
		{Name: "empty", Kind: ifevalEndRule},
	} {
		if _, err := compileInstructionRule(c); err == nil {
			t.Errorf("unbound native grammar accepted: %+v", c)
		}
	}
}
