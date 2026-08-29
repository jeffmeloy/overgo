package evaluation

import (
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodb"
)

// TestHighFanoutReadsAreBatched proves shard-report verification does
// not reopen storage once per observation: resuming a completed
// campaign reads its outputs through batched visits, and per-item
// opens stay below the case count instead of scaling with it.
func TestHighFanoutReadsAreBatched(t *testing.T) {
	ctx := t.Context()
	suite := exactFixture()
	suite.Cases = nil
	for ordinal := range 8 {
		suite.Cases = append(suite.Cases, ExactCase{
			Name: fmt.Sprintf("case-%d", ordinal), Prompt: fmt.Sprintf("prompt-%d", ordinal),
			MaxTokens: exactMaxTokens, Text: "ok",
			PromptTokens: exactPromptTokens, GeneratedTokens: exactGeneratedTokens,
		})
	}
	exact, err := CompileExact(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "batched-model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "batched-recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "batched-environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleIsolated},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, storeErr := overgodb.Open(t.TempDir())
	if storeErr != nil {
		t.Fatal(storeErr)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	generator := &countingExactGenerator{}
	if _, err := EvaluateExactSharded(ctx, store, generator, exact, plan, nil); err != nil {
		t.Fatal(err)
	}
	counting := &repositorytest.CountingRepository{Repository: store}
	if _, err := EvaluateExactSharded(ctx, counting, generator, exact, plan, nil); err != nil {
		t.Fatal(err)
	}
	cases := len(exact.suite.Cases)
	if cases < 2 {
		t.Fatalf("fixture suite too small to observe scaling: %d cases", cases)
	}
	if counting.Batches == 0 {
		t.Fatal("resume verification performed no batched reads")
	}
	if counting.Opens >= cases {
		t.Fatalf("per-item opens %d scale with the %d-case suite", counting.Opens, cases)
	}
}
