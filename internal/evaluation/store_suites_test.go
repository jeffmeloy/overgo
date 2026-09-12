package evaluation

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func rawFields(t *testing.T, pairs map[string]any) map[string]json.RawMessage {
	t.Helper()
	fields := make(map[string]json.RawMessage, len(pairs))
	for name, value := range pairs {
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		fields[name] = encoded
	}
	return fields
}

// TestStoreSuiteAssemblers pins each family's derivation convention
// against the shapes the port's own fixtures use: lm_eval records
// become exactly the suite cases the suite compilers admit.
func TestStoreSuiteAssemblers(t *testing.T) {
	mmlu, _, err := assembleMMLUSuite([]storeCase{{
		entry: "mmlu/abstract_algebra/test", subset: "abstract_algebra", ordinal: 0,
		fields: rawFields(t, map[string]any{
			"question": "Which number is even?", "choices": []string{"three", "four"}, "answer": 1,
		}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	mmluSuite := mmlu.(MultipleChoiceSuite)
	if mmluSuite.Cases[0].Prompt != "Which number is even?\nA. three\nB. four\nAnswer:" ||
		mmluSuite.Cases[0].Candidates[1] != " B" || mmluSuite.Cases[0].Answer != 1 {
		t.Fatalf("mmlu case = %+v", mmluSuite.Cases[0])
	}

	bbh, _, err := assembleChoiceGroups("bbh", "lm-eval/leaderboard-bbh/v1.0", []storeCase{
		{entry: "bbh/boolean_expressions/test", subset: "boolean_expressions", ordinal: 0,
			fields: rawFields(t, map[string]any{"input": "not True is", "target": "False"})},
		{entry: "bbh/boolean_expressions/test", subset: "boolean_expressions", ordinal: 1,
			fields: rawFields(t, map[string]any{"input": "True or False is", "target": "True"})},
	})
	if err != nil {
		t.Fatal(err)
	}
	bbhSuite := bbh.(GroupedChoiceSuite)
	if bbhSuite.Cases[0].Prompt != "Q: not True is\nA:" ||
		len(bbhSuite.Cases[0].Candidates) != 2 || bbhSuite.Cases[0].Candidates[0] != " False" ||
		bbhSuite.Cases[0].Answer != 0 || bbhSuite.Cases[1].Answer != 1 {
		t.Fatalf("bbh cases = %+v", bbhSuite.Cases)
	}

	musr, _, err := assembleMuSRSuite([]storeCase{{
		entry: "musr/default/object_placements", subset: "object_placements", ordinal: 0,
		fields: rawFields(t, map[string]any{
			"narrative": "A is left of B.", "question": "Where is A?",
			"choices": `['left', 'right']`, "answer_index": 0,
		}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	musrSuite := musr.(GroupedChoiceSuite)
	if musrSuite.Cases[0].Prompt != "A is left of B.\n\nWhere is A?\n\n1 - left\n2 - right\nAnswer:" ||
		musrSuite.Cases[0].Candidates[0] != " left" {
		t.Fatalf("musr case = %+v", musrSuite.Cases[0])
	}

	math, dropped, err := assembleMATHSuite([]storeCase{
		{entry: "math/algebra/test", subset: "algebra", ordinal: 0,
			fields: rawFields(t, map[string]any{
				"problem": "Divide one by two.", "solution": `Work gives $\boxed{\frac{1}{2}}$.`,
				"level": "Level 1", "type": "Algebra",
			})},
		{entry: "math/algebra/test", subset: "algebra", ordinal: 1,
			fields: rawFields(t, map[string]any{
				"problem": "No box here.", "solution": "unboxed", "level": "Level 1", "type": "Algebra",
			})},
	})
	if err != nil || dropped != 1 {
		t.Fatalf("math assemble = (%v, dropped=%d)", err, dropped)
	}
	mathSuite := math.(StructuredGeneratedSuite)
	if mathSuite.Cases[0].Prompt != "Problem: Divide one by two.\nAnswer:" ||
		mathSuite.Cases[0].Answers[0] != `\frac{1}{2}` {
		t.Fatalf("math case = %+v", mathSuite.Cases[0])
	}

	ifeval, dropped, err := assembleIFEvalSuite([]storeCase{
		{entry: "ifeval/default/train", subset: "default", ordinal: 0,
			fields: rawFields(t, map[string]any{
				"key": 1, "prompt": "Reply in uppercase and include HELLO.",
				"instruction_id_list": []string{"change_case:english_capital", "keywords:existence"},
				"kwargs":              []map[string]any{{}, {"keywords": []string{"HELLO"}}},
			})},
		{entry: "ifeval/default/train", subset: "default", ordinal: 1,
			fields: rawFields(t, map[string]any{
				"key": 2, "prompt": "Respond in French.",
				"instruction_id_list": []string{"language:response_language"},
				"kwargs":              []map[string]any{{"language": "fr"}},
			})},
	})
	if err != nil || dropped != 1 {
		t.Fatalf("ifeval assemble = (%v, dropped=%d)", err, dropped)
	}
	ifevalSuite := ifeval.(InstructionRulesSuite)
	if len(ifevalSuite.Cases) != 1 || len(ifevalSuite.Cases[0].Rules) != 2 ||
		ifevalSuite.Cases[0].Rules[0].Kind != RuleUppercase ||
		ifevalSuite.Cases[0].Rules[1].Kind != ifevalKeywordsRule {
		t.Fatalf("ifeval case = %+v", ifevalSuite.Cases)
	}
}

// TestDeriveStoreSuites pins the wiring end to end: an imported cache
// publishes the catalog, and derivation compiles a runnable suite from
// the store alone.
func TestDeriveStoreSuites(t *testing.T) {
	fixture, err := os.ReadFile(filepath.Join("..", "dataset", "testdata", "mmlu-dev.arrow"))
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	target := filepath.Join(root, "cais___mmlu", "abstract_algebra", "0.0.0", "aa11", "mmlu-test.arrow")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, fixture, 0o644); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, _, err := CatalogHFCacheBenchmarks(ctx, store, root); err != nil {
		t.Fatal(err)
	}
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution:   ExecutionPolicy{Lifecycle: LifecycleResident},
	}
	suites, skipped, err := DeriveStoreSuites(ctx, store, authorities)
	if err != nil {
		t.Fatal(err)
	}
	if len(suites) != 1 || len(skipped) != 0 {
		t.Fatalf("derived = %d suites, skipped %v", len(suites), skipped)
	}
	descriptor := suites[0].Descriptor()
	if descriptor.Kind != MultipleChoiceKind || descriptor.Cases != 5 {
		t.Fatalf("descriptor = %+v, want the 5-case multiple-choice suite", descriptor)
	}
	// A family request compiles that family alone and names an absent
	// family instead of deriving nothing silently.
	only, _, err := DeriveStoreSuiteFamily(ctx, store, authorities, "mmlu")
	if err != nil || len(only) != 1 || only[0].Descriptor().Source != "store/mmlu" {
		t.Fatalf("family derivation = %d suites, %v", len(only), err)
	}
	if _, _, err := DeriveStoreSuiteFamily(ctx, store, authorities, "bbh"); err == nil ||
		!strings.Contains(err.Error(), "no bbh family") {
		t.Fatalf("absent family error = %v", err)
	}
}

// TestAssembleDNASuite pins the corpus-slice window: whole-genome rows
// truncate to the fixed scoring window every model scores identically,
// and rows inside the window pass through whole.
func TestAssembleDNASuite(t *testing.T) {
	long := strings.Repeat("ACGTCA", dnaScoringWindowChars/6+7)
	assembled, dropped, err := assembleDNASuite([]storeCase{
		{
			entry: "dna/mrna_evo2/corpus-slice", subset: "mrna_evo2", ordinal: 0,
			fields: rawFields(t, map[string]any{"text": long}),
		},
		{
			entry: "dna/mrna_evo2/corpus-slice", subset: "mrna_evo2", ordinal: 1,
			fields: rawFields(t, map[string]any{"text": "ACGT"}),
		},
	})
	if err != nil || dropped != 0 {
		t.Fatalf("assemble = (%v, %d)", err, dropped)
	}
	suite := assembled.(SequenceScoringSuite)
	if len(suite.Cases) != 2 || len(suite.Cases[0].Text) != dnaScoringWindowChars ||
		suite.Cases[0].Text != long[:dnaScoringWindowChars] || suite.Cases[1].Text != "ACGT" {
		t.Fatalf("window = %d and %d chars, want %d and 4",
			len(suite.Cases[0].Text), len(suite.Cases[1].Text), dnaScoringWindowChars)
	}
	if suite.Cases[0].Name != "dna/mrna_evo2/corpus-slice/0" || suite.Cases[0].Group != "mrna_evo2" {
		t.Fatalf("case identity = %+v", suite.Cases[0])
	}
}
