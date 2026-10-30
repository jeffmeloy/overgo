package evaluation

import (
	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"testing"
)

func TestQwenFourRetainedBenchmarkAcceptance(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	parse := func(s string) artifact.ID {
		id, e := artifact.ParseID(s)
		if e != nil {
			t.Fatal(e)
		}
		return id
	}
	// The conversion's activation is retired since the 4B is served from its own files
	// (RetireSupersededActivation); its retained evidence binds to the recipe it was activated under.
	activated, err := recipe.RequireDefinition(t.Context(), store, parse("recipe:sha256:49d1ed7c41e45abcc9c10ff13ec247ab775f7054123d5507caa1f63c0a19f803"))
	if err != nil {
		t.Fatal(err)
	}
	if model, found := activated.PrimaryDependency(recipe.DependencyModel); !found || model != parse("model:sha256:887f41243686e101a314013a168b6bad5df19f8bee5df3e45d942ff79455413b") {
		t.Fatal("the recipe is not the conversion's")
	}
	definition, found := activated.PrimaryDependency(recipe.DependencyDefinition)
	if !found {
		t.Fatal("definition missing")
	}
	value, err := RequireEvaluationEvidence(t.Context(), store, parse("evidence:sha256:c8cb148849f1c3d030094d7d1ed44c89529f4a910ca3a8ee69a6f8236341a43d"))
	if err != nil {
		t.Fatal(err)
	}
	if value.Recipe != activated.ID || value.ModelDefinition != definition {
		t.Fatal("activation identity differs")
	}
	var body planBody
	readRetainedEvidence(t, store, value.Plan, &body)
	var suite MultipleChoiceSuite
	readRetainedEvidence(t, store, body.CaseProfile, &suite)
	var execution ExecutionPolicy
	readRetainedEvidence(t, store, body.Execution, &execution)
	compiled, err := CompileMultipleChoice(suite)
	if err != nil {
		t.Fatal(err)
	}
	bound, err := BindMultipleChoice(compiled, ExactAuthorities{ModelDefinition: definition, RuntimeRecipe: value.Recipe, Environment: value.Environment, CodeCommit: value.CodeCommit, Execution: execution})
	if err != nil || bound.Identity() != value.Plan {
		t.Fatalf("binding differs: %v", err)
	}
	var stored struct {
		GroupedChoiceReport
		Categories []accuracyGroup
	}
	readRetainedEvidence(t, store, value.Report, &stored)
	stored.Groups = stored.Categories
	if len(suite.Cases) != 12032 || len(stored.Categories) != 14 || execution.Prompting != PromptingChatTemplate {
		t.Fatal("denominator or prompting differs")
	}
	if err := checkRetainedGeneratedChoices(suite.Cases, stored.GroupedChoiceReport, 5617); err != nil {
		t.Fatal(err)
	}
	corrupt := stored.GroupedChoiceReport
	corrupt.Observations = corrupt.Observations[1:]
	if checkRetainedGeneratedChoices(suite.Cases, corrupt, 5617) == nil {
		t.Fatal("accepted an omitted retained answer")
	}
	t.Log("5617/12032 generated answers; 14 categories; original plan and current extractor agree; no model run")
}
