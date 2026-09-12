package evaluation

import (
	"slices"
	"testing"
)

// lm_eval 0.4.9.1, ifeval.yaml task v4.0: max_gen_toks=1280.
// instructions.py SHA256 1556285b56cb81a7a1ada79327cd8797681aed004160c7f87b3a94f3e10bae32.
func TestIFEvalProtocolAcceptance(t *testing.T) {
	assembled, dropped, err := assembleIFEvalSuite([]storeCase{{
		entry: "ifeval/default/train", subset: "default", ordinal: 0,
		fields: rawFields(t, map[string]any{
			"prompt": "Quote your answer.", "instruction_id_list": []string{"startend:quotation"},
			"kwargs": []map[string]any{{}},
		}),
	}})
	if err != nil || dropped != 0 {
		t.Fatalf("assemble: dropped=%d err=%v", dropped, err)
	}
	suite := assembled.(InstructionRulesSuite)
	if len(suite.Cases) != 1 || suite.Cases[0].MaxTokens != 1280 || len(suite.Cases[0].Rules) != 1 {
		t.Fatalf("native cap and instruction denominator: %+v", suite.Cases)
	}
	compiled, err := CompileInstructionRules(suite)
	if err != nil {
		t.Fatal(err)
	}
	// Captured strict/loose answers from the pinned native QuotationChecker.
	for _, c := range []struct {
		response      string
		strict, loose bool
	}{
		{`"hello"`, true, true},
		{" \"hello\" \n", true, true},
		{`"`, false, false},
		{`""`, true, true},
		{"hello", false, false},
		{"\"first\nsecond\"", true, true},
		{"header\n\"hello\"\nfooter", false, true},
	} {
		strict, loose, err := evaluateInstructionViews(c.response, compiled.rules[0])
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(strict, []bool{c.strict}) || !slices.Equal(loose, []bool{c.loose}) {
			t.Errorf("response=%q strict=%v loose=%v want=%t/%t", c.response, strict, loose, c.strict, c.loose)
		}
	}
	// Both corrections create new evaluator identities; old reports stay immutable.
	oldCap := suite
	oldCap.Cases = slices.Clone(suite.Cases)
	oldCap.Cases[0].MaxTokens = 256
	oldRules := suite
	oldRules.Cases = slices.Clone(suite.Cases)
	oldRules.Cases[0].Rules = []InstructionRule{
		{Name: "startend:quotation/open", Kind: RulePrefix, Values: []string{`"`}},
		{Name: "startend:quotation/close", Kind: RuleSuffix, Values: []string{`"`}},
	}
	for _, old := range []InstructionRulesSuite{oldCap, oldRules} {
		prior, err := CompileInstructionRules(old)
		if err != nil {
			t.Fatal(err)
		}
		if prior.identity == compiled.identity || prior.dataset == compiled.dataset || prior.split == compiled.split {
			t.Fatal("protocol/scorer correction reused old evaluator lineage")
		}
	}
}
