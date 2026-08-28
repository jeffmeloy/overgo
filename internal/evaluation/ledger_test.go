package evaluation

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
)

type countingExactGenerator struct {
	calls int
}

func (g *countingExactGenerator) Generate(
	ctx context.Context,
	prompt string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	g.calls++
	return (exactGenerator{pieces: []string{"o", "k"}}).Generate(ctx, prompt, options)
}

func TestResumeRejectsPartialOrForeignPlanResults(t *testing.T) {
	ctx := context.Background()
	exact, plan := ledgerFixture(t, "environment")

	t.Run("resume", func(t *testing.T) {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		publishPlanFixtureAuthorities(t, store, plan)
		generator := &countingExactGenerator{}
		first, err := EvaluateExactSharded(ctx, store, generator, exact, plan, nil)
		if err != nil {
			t.Fatal(err)
		}
		second, err := EvaluateExactSharded(ctx, store, generator, exact, plan, nil)
		if err != nil {
			t.Fatal(err)
		}
		if first != second || generator.calls != len(exact.suite.Cases) {
			t.Fatalf("resume report/calls = %s/%s/%d", first, second, generator.calls)
		}
		otherExact, otherPlan := ledgerFixture(t, "other-environment")
		publishPlanFixtureAuthorities(t, store, otherPlan)
		if _, err := EvaluateExactSharded(ctx, store, generator, otherExact, otherPlan, nil); err != nil {
			t.Fatal(err)
		}
		if generator.calls != len(exact.suite.Cases)+len(otherExact.suite.Cases) {
			t.Fatalf("changed authority reused prior shards: calls=%d", generator.calls)
		}
	})

	t.Run("partial", func(t *testing.T) {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		shards, err := compileExactShards(exact, plan)
		if err != nil {
			t.Fatal(err)
		}
		missing := planID(t, artifact.KindEvaluation, "missing-report")
		alias := shardAlias(plan.identity, shards[0].ID)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: alias, Artifacts: []artifact.Descriptor{{ID: missing}},
			Aliases: []artifact.AliasBinding{{Name: alias, Target: missing}},
		}); err != nil {
			t.Fatal(err)
		}
		generator := &countingExactGenerator{}
		if _, err := EvaluateExactSharded(ctx, store, generator, exact, plan, nil); err == nil || generator.calls != 0 {
			t.Fatalf("partial report accepted or executed: calls=%d err=%v", generator.calls, err)
		}
	})

	t.Run("foreign", func(t *testing.T) {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		shards, err := compileExactShards(exact, plan)
		if err != nil {
			t.Fatal(err)
		}
		_, foreignPlan := ledgerFixture(t, "foreign-environment")
		foreignShard := shards[0]
		foreignShard.Plan = foreignPlan.identity
		foreignShard.ID = planID(t, artifact.KindDatasetShard, "foreign-shard")
		foreignOutput := planID(t, artifact.KindOutput, "foreign-output")
		foreignReport, err := newShardReport(
			foreignShard,
			[]caseObservation{{Case: foreignShard.Cases[0], Output: foreignOutput, Passed: true}},
			[]metricState{{Name: exactMetricName, Sum: 1, Count: 1}},
		)
		if err != nil {
			t.Fatal(err)
		}
		content, err := shardReportContract.ContentJSON(foreignReport.ID, foreignReport)
		if err != nil {
			t.Fatal(err)
		}
		alias := shardAlias(plan.identity, shards[0].ID)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: alias, Contents: []artifact.Content{content},
			Aliases: []artifact.AliasBinding{{Name: alias, Target: foreignReport.ID}},
		}); err != nil {
			t.Fatal(err)
		}
		generator := &countingExactGenerator{}
		if _, err := EvaluateExactSharded(ctx, store, generator, exact, plan, nil); err == nil || generator.calls != 0 {
			t.Fatalf("foreign report accepted or executed: calls=%d err=%v", generator.calls, err)
		}
	})
}

func ledgerFixture(t testing.TB, environment string) (ExactPlan, Plan) {
	t.Helper()
	exact, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, environment),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleIsolated},
	})
	if err != nil {
		t.Fatal(err)
	}
	return exact, plan
}
