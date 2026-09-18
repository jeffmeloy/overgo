package evaluation

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
)

type countingExactGenerator struct {
	calls   int
	results map[string]exactGenerator
}

func (g *countingExactGenerator) Generate(
	ctx context.Context,
	prompt string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	g.calls++
	if result, ok := g.results[prompt]; ok {
		return result.Generate(ctx, prompt, options)
	}
	return (exactGenerator{pieces: []string{"o", "k"}}).Generate(ctx, prompt, options)
}

func TestExactFailureRetention(t *testing.T) {
	for _, interruption := range []string{"none", "observer", "cancel", "generation"} {
		t.Run(interruption, func(t *testing.T) {
			suite := exactFixture()
			second := suite.Cases[0]
			second.Name, second.Prompt = "second", "second prompt"
			suite.Cases = append(suite.Cases, second)
			exact, err := CompileExact(suite)
			if err != nil {
				t.Fatal(err)
			}
			_, base := ledgerFixture(t, "retained-failure")
			plan, err := BindExact(exact, ExactAuthorities{
				ModelDefinition: base.body.ModelDefinition, RuntimeRecipe: base.body.RuntimeRecipe,
				CodeCommit: base.body.CodeCommit, Environment: base.body.Environment,
				Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
			})
			if err != nil {
				t.Fatal(err)
			}
			root := t.TempDir()
			store, err := overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { store.Close() }()
			publishPlanFixtureAuthorities(t, store, plan)
			generator := &countingExactGenerator{results: map[string]exactGenerator{
				suite.Cases[0].Prompt: {pieces: []string{"n", "o"}},
			}}
			stop := errors.New("interrupted")
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			var observe func(ExactResult) error
			switch interruption {
			case "observer":
				observe = func(ExactResult) error { return stop }
			case "cancel":
				observe = func(ExactResult) error { cancel(context.Canceled); return nil }
			case "generation":
				generator.results[second.Prompt] = exactGenerator{err: stop}
			}
			first, err := EvaluateExactSharded(ctx, store, generator, exact, plan, observe)
			switch interruption {
			case "none":
				if !first.Valid() || !errors.Is(err, errExactMismatch) {
					t.Fatalf("completed mismatch: %s %v", first, err)
				}
			case "cancel":
				if first.Valid() || !errors.Is(err, context.Canceled) {
					t.Fatalf("cancellation: %s %v", first, err)
				}
			default:
				if first.Valid() || !errors.Is(err, stop) {
					t.Fatalf("interruption: %s %v", first, err)
				}
			}
			shards, err := compileExactShards(exact, plan)
			if err != nil {
				t.Fatal(err)
			}
			stored, err := loadShardReports(t.Context(), store, plan, shards)
			if err != nil {
				t.Fatal(err)
			}
			failed, ok := stored[0]
			if !ok || failed.Observations[0].Passed || failed.Observations[0].Failure == "" || failed.Metrics[0].Sum != 0 || failed.Metrics[0].Count != 1 {
				t.Fatalf("failed acquisition not retained: %+v", stored)
			}
			if interruption != "none" && len(stored) != 1 {
				t.Fatalf("unfinished case cached: %+v", stored)
			}
			calls := generator.calls
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			store, err = overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			delete(generator.results, second.Prompt)
			resumed, err := EvaluateExactSharded(t.Context(), store, generator, exact, plan, nil)
			if !resumed.Valid() || !errors.Is(err, errExactMismatch) {
				t.Fatalf("resume lost failure: %s %v", resumed, err)
			}
			missing := len(shards) - len(stored)
			if generator.calls != calls+missing {
				t.Fatalf("repeated acquisition: calls=%d want=%d", generator.calls, calls+missing)
			}
			stored, err = loadShardReports(t.Context(), store, plan, shards)
			if err != nil {
				t.Fatal(err)
			}
			if len(stored) != len(shards) || !stored[1].Observations[0].Passed {
				t.Fatalf("independent case did not finish: %+v", stored)
			}
			calls = generator.calls
			replayed, err := EvaluateExactSharded(t.Context(), store, generator, exact, plan, nil)
			if replayed != resumed || !errors.Is(err, errExactMismatch) || generator.calls != calls {
				t.Fatalf("failed replay regenerated or passed: %s %v calls=%d", replayed, err, generator.calls)
			}
		})
	}
}

func TestResumeRejectsPartialOrForeignPlanResults(t *testing.T) {
	ctx := t.Context()
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
