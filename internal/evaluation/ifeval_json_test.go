package evaluation

import (
	_ "embed"
	"encoding/json"
	"slices"
	"testing"
)

//go:embed testdata/ifeval_json_oracle.json
var ifevalJSONOracle []byte

func TestIFEvalJSONAcceptance(t *testing.T) {
	var oracle struct {
		ReferenceVersion string            `json:"reference_version"`
		Sources          map[string]string `json:"sources"`
		Cases            []struct {
			Response string `json:"response"`
			Strict   bool   `json:"strict"`
			Loose    bool   `json:"loose"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(ifevalJSONOracle, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.ReferenceVersion != "0.4.9.1" || len(oracle.Cases) != 66 ||
		oracle.Sources["instructions.py"] != "1556285b56cb81a7a1ada79327cd8797681aed004160c7f87b3a94f3e10bae32" ||
		oracle.Sources["utils.py"] != "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632" {
		t.Fatal("native checker identity or denominator differs")
	}
	assembled, dropped, err := assembleIFEvalSuite([]storeCase{{
		entry: "ifeval/default/train", subset: "default", ordinal: 0,
		fields: rawFields(t, map[string]any{
			"prompt": "Return JSON.", "instruction_id_list": []string{"detectable_format:json_format"},
			"kwargs": []map[string]any{{}},
		}),
	}})
	if err != nil || dropped != 0 {
		t.Fatalf("assemble: dropped=%d err=%v", dropped, err)
	}
	suite := assembled.(InstructionRulesSuite)
	compiled, err := CompileInstructionRules(suite)
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.rules) != 1 || len(compiled.rules[0]) != 1 {
		t.Fatal("one native instruction must produce one metric entry")
	}
	for index, c := range oracle.Cases {
		strict, loose := evaluateInstructionViews(c.Response, compiled.rules[0])
		if !slices.Equal(strict, []bool{c.Strict}) || !slices.Equal(loose, []bool{c.Loose}) {
			t.Errorf("case %d response=%q strict=%v loose=%v want=%t/%t", index, c.Response, strict, loose, c.Strict, c.Loose)
		}
	}
	priorSuite := suite
	priorSuite.Cases = slices.Clone(suite.Cases)
	priorSuite.Cases[0].Rules = []InstructionRule{{Name: "detectable_format:json_format", Kind: RuleJSON}}
	prior, err := CompileInstructionRules(priorSuite)
	if err != nil {
		t.Fatal(err)
	}
	if prior.identity == compiled.identity || prior.dataset == compiled.dataset || prior.split == compiled.split {
		t.Fatal("changed scorer reused the old evaluator identity")
	}
	for _, response := range []string{"NaN", "Infinity", "-Infinity", "```json\n{}\n```"} {
		if prior.rules[0][0].matches(response) {
			t.Errorf("strict JSON rule now accepts %q", response)
		}
	}
	if !prior.rules[0][0].matches(`{"NaN":"Infinity"}`) {
		t.Fatal("strict JSON string semantics changed")
	}
	for _, invalid := range []InstructionRule{
		{Name: "invalid", Kind: ifevalJSONRule, Values: []string{"unexpected"}},
		{Name: "invalid", Kind: ifevalJSONRule, Count: &CountRule{Relation: RelationEqual, Value: 1}},
	} {
		if _, err := compileInstructionRule(invalid); err == nil {
			t.Fatal("JSON checker accepted parameters absent from the native contract")
		}
	}
}
