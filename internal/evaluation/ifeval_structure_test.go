package evaluation

import (
	_ "embed"
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
)

//go:embed testdata/ifeval_structure_oracle.json
var ifevalStructureOracle []byte

func TestIFEvalStructureContractAcceptance(t *testing.T) {
	t.Run("compiled criteria own their input values", func(t *testing.T) {
		suite := InstructionRulesSuite{Kind: InstructionRulesKind, Schema: "native-count-ownership", Source: "fixture", Cases: []InstructionRulesCase{{Name: "count", Prompt: "Repeat cat.", MaxTokens: 1, Rules: []InstructionRule{{Name: "frequency", Kind: ifevalFrequency, Values: []string{"cat"}, Count: &CountRule{Relation: RelationAtLeast, Value: 2}}}}}}
		compiled, err := CompileInstructionRules(suite)
		if err != nil {
			t.Fatal(err)
		}
		suite.Cases[0].Rules[0].Values[0] = "dog"
		suite.Cases[0].Rules[0].Count.Value = 3
		strict, loose := evaluateInstructionViews("catcat", compiled.rules[0])
		if !slices.Equal(strict, []bool{true}) || !slices.Equal(loose, []bool{true}) {
			t.Fatal("caller mutation changed compiled criteria")
		}
		identity, err := artifact.JSONID(artifact.KindProfile, compiled.suite)
		if err != nil || identity != compiled.identity {
			t.Fatalf("caller mutation changed published criteria: %s %v", identity, err)
		}
	})
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
	if err := json.Unmarshal(ifevalStructureOracle, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.ReferenceVersion != "0.4.9.1" || len(oracle.Cases) != 131 || oracle.Sources["instructions.py"] != "1556285b56cb81a7a1ada79327cd8797681aed004160c7f87b3a94f3e10bae32" || oracle.Sources["instructions_util.py"] != "e8c4d9187bac1482d93941fb46469609c3ae78d896195bfcb300a0707d492567" || oracle.Sources["utils.py"] != "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632" {
		t.Fatal("native source or denominator changed")
	}
	families := map[string]bool{}
	for index, c := range oracle.Cases {
		assembled, dropped, err := assembleIFEvalSuite([]storeCase{{entry: "ifeval/default/train", subset: "default", ordinal: index, fields: rawFields(t, map[string]any{"prompt": "Follow the stated instruction.", "instruction_id_list": []string{c.Instruction}, "kwargs": []map[string]json.RawMessage{c.Kwargs}})}})
		if err != nil || dropped != 0 {
			t.Fatalf("%d %s: dropped=%d error=%v", index, c.Instruction, dropped, err)
		}
		compiled, err := CompileInstructionRules(assembled.(InstructionRulesSuite))
		if err != nil {
			t.Fatal(err)
		}
		if len(compiled.rules) != 1 || len(compiled.rules[0]) != 1 {
			t.Fatal("instruction denominator expanded")
		}
		strict, loose := evaluateInstructionViews(c.Response, compiled.rules[0])
		if !slices.Equal(strict, c.Strict) || !slices.Equal(loose, c.Loose) {
			t.Errorf("%d %s %q: strict=%v/%v loose=%v/%v", index, c.Instruction, c.Response, strict, c.Strict, loose, c.Loose)
		}
		families[c.Instruction] = true
	}
	if len(families) != 12 {
		t.Fatalf("families=%d", len(families))
	}
	for _, c := range []struct {
		id     string
		kwargs map[string]any
	}{
		{"length_constraints:number_words", map[string]any{"num_words": 0, "relation": "less than"}},
		{"length_constraints:number_words", map[string]any{"num_words": -1, "relation": "at least"}},
		{"length_constraints:number_words", map[string]any{"num_words": 2.5, "relation": "at least"}},
		{"length_constraints:number_words", map[string]any{"num_words": 3, "relation": "at most"}},
		{"keywords:frequency", map[string]any{"keyword": "a|b", "frequency": 2, "relation": "at least"}},
		{"keywords:frequency", map[string]any{"keyword": " ", "frequency": 2, "relation": "at least"}},
		{"detectable_format:multiple_sections", map[string]any{"section_spliter": "S.*", "num_sections": 2}},
		{"detectable_content:postscript", map[string]any{"postscript_marker": "[PS]"}},
		{"combination:repeat_prompt", map[string]any{"prompt_to_repeat": "Τέλος"}},
		{"keywords:letter_frequency", map[string]any{"letter": "#", "let_frequency": 4, "let_relation": "at least"}},
	} {
		if _, ok := ifevalRuleFor(c.id, rawFields(t, c.kwargs)); ok {
			t.Errorf("unresolved native parameters accepted: %s %v", c.id, c.kwargs)
		}
	}
}
