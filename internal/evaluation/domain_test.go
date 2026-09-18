package evaluation

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestDomainRoutedSuites pins the routing contract: a declared domain
// filters suites to the ones its scores are meaningful in, an
// undeclared model keeps full coverage, and re-declaration replaces
// the binding.
func TestDomainRoutedSuites(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "domain-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/domain/model", Artifacts: []artifact.Descriptor{{ID: model}},
	}); err != nil {
		t.Fatal(err)
	}

	if _, declared, err := modelrecipe.EvalDomains(ctx, store, model); err != nil || declared {
		t.Fatalf("undeclared model = (%t, %v)", declared, err)
	}
	if _, err := modelrecipe.DeclareEvalDomains(ctx, store, model, []string{"dna"}); err != nil {
		t.Fatal(err)
	}
	domains, declared, err := modelrecipe.EvalDomains(ctx, store, model)
	if err != nil || !declared || len(domains) != 1 || domains[0] != "dna" {
		t.Fatalf("declared domains = (%v, %t, %v)", domains, declared, err)
	}
	if _, err := modelrecipe.DeclareEvalDomains(ctx, store, model, []string{"dna", "text"}); err != nil {
		t.Fatal(err)
	}
	if domains, _, _ := modelrecipe.EvalDomains(ctx, store, model); len(domains) != 2 {
		t.Fatalf("redeclared domains = %v", domains)
	}

	if SuiteDomain("store/mmlu") != DomainText || SuiteDomain("store/unknown") != DomainText {
		t.Fatal("suite domains must default to text")
	}

	text := compiledSuiteWithSource(t, "store/mmlu")
	all := []CompiledSuite{text}
	if kept := FilterSuitesForDomains(all, nil, false); len(kept) != 1 {
		t.Fatalf("undeclared filter kept %d", len(kept))
	}
	if kept := FilterSuitesForDomains(all, []string{"dna"}, true); len(kept) != 0 {
		t.Fatalf("dna-only model kept %d text suites", len(kept))
	}
	if kept := FilterSuitesForDomains(all, []string{"dna", DomainText}, true); len(kept) != 1 {
		t.Fatalf("dna+text model kept %d", len(kept))
	}
}

func compiledSuiteWithSource(t *testing.T, source string) CompiledSuite {
	t.Helper()
	suite, _, err := assembleMMLUSuite([]storeCase{{
		entry: "mmlu/abstract_algebra/test", subset: "abstract_algebra", ordinal: 0,
		fields: rawFields(t, map[string]any{
			"question": "Which number is even?", "choices": []string{"three", "four"}, "answer": 1,
		}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	shaped := suite.(MultipleChoiceSuite)
	shaped.Source = source
	data, err := json.Marshal(shaped)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileSuite(data, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
		Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution:   ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	return compiled
}
