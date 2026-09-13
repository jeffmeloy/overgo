package evaluation

import (
	"slices"
	"testing"
)

func TestIFEvalRepeatedInstructionOccurrences(t *testing.T) {
	// Two instances of one family carry different operands; neither is redundant.
	const family = "keywords:existence"
	entry := storeCase{entry: "ifeval/default/train", fields: rawFields(t, map[string]any{
		"prompt": "Include alpha and beta.", "instruction_id_list": []string{family, family},
		"kwargs": []map[string]any{{"keywords": []string{"alpha"}}, {"keywords": []string{"beta"}}},
	})}
	assembled, dropped, err := assembleIFEvalSuite([]storeCase{entry})
	if err != nil || dropped != 0 {
		t.Fatalf("assemble: dropped=%d err=%v", dropped, err)
	}
	suite := assembled.(InstructionRulesSuite)
	compiled, err := CompileInstructionRules(suite)
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.rules[0]) != 2 {
		t.Fatal("repeated instruction disappeared")
	}
	for _, c := range []struct {
		response string
		want     []bool
	}{
		{"alpha", []bool{true, false}}, {"beta", []bool{false, true}}, {"alpha beta", []bool{true, true}},
	} {
		strict, loose, err := evaluateInstructionViews(c.response, compiled.rules[0])
		if err != nil || !slices.Equal(strict, c.want) || !slices.Equal(loose, c.want) {
			t.Fatalf("%q: strict=%v loose=%v err=%v", c.response, strict, loose, err)
		}
	}
	// The original single occurrence keeps its identity spelling.
	if suite.Cases[0].Rules[0].Name != family || suite.Cases[0].Rules[1].Name == family {
		t.Fatal("instruction occurrence names differ")
	}
	repeated, _, err := assembleIFEvalSuite([]storeCase{entry})
	if err != nil {
		t.Fatal(err)
	}
	replay, err := CompileInstructionRules(repeated.(InstructionRulesSuite))
	if err != nil || replay.identity != compiled.identity {
		t.Fatal("compilation is not deterministic")
	}
}
