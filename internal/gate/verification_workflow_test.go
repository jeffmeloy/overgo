package gate

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/runrecord"
)

// TestAnalysisComposedVerification composes resolved impact obligations into a
// deterministic verification DAG and runs it through the shared evidence cache.
// It proves the composition orders a cheap contract check ahead of the static
// analysis it gates and the expensive test behind that, runs each obligation
// once, reuses every unchanged node when the same obligations recompose under a
// different candidate tree, isolates a failed obligation to its dependents while
// an independent lane still runs, and refuses a dependency cycle.
func TestAnalysisComposedVerification(t *testing.T) {
	t.Parallel()
	base, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("base"))
	environment, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("environment"))
	input, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("obligation inputs"))
	source := strings.Repeat("a", 64)
	surface := automationcheck.Surface{Identity: "base:candidate"}

	// One obligation set: contract -> static -> test, plus an independent lane.
	// Each obligation's runner counts its executions and fails the named one.
	buildChecks := func(runs map[string]int, failing string) []automationcheck.Check {
		build := func(name string, deps []string) automationcheck.Check {
			return automationcheck.Check{
				Descriptor: automationcheck.Descriptor{Name: name, Phase: runrecord.PhaseTest, Always: true, Dependencies: deps},
				Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
					runs[name]++
					if name == failing {
						return false, "", errors.New("seeded failure")
					}
					return false, "", nil
				},
			}
		}
		return []automationcheck.Check{
			build("contract", nil),
			build("static", []string{"contract"}),
			build("test", []string{"static"}),
			build("lane", nil),
		}
	}

	// compose binds every planned obligation of one composition to an exact plan
	// identity and explicit inputs, the way the gate composes a candidate's DAG.
	compose := func(t *testing.T, tree string, runs map[string]int, failing string) []automationcheck.Invocation {
		t.Helper()
		invocations, err := automationcheck.Plan(buildChecks(runs, failing), automationcheck.Impact{})
		if err != nil {
			t.Fatal(err)
		}
		candidate, _ := artifact.IdentifyBytes(artifact.KindProfile, []byte("candidate-"+tree))
		plan, err := automationcheck.BindManifestPlan(base, candidate, source, tree, surface, automationcheck.Impact{}, invocations)
		if err != nil {
			t.Fatal(err)
		}
		bound := make([]automationcheck.Invocation, len(invocations))
		for i, invocation := range invocations {
			execution, err := automationcheck.BindManifestExecution(plan, invocation, []artifact.ID{input},
				&automationcheck.ReuseBinding{Input: input, Environment: environment})
			if err != nil {
				t.Fatal(err)
			}
			bound[i] = execution
		}
		return bound
	}

	// A cache-backed executor: one mutex serializes the cache and the counting
	// runner it drives, so a concurrent wave never races either.
	newExecutor := func(cache *automationcheck.EvidenceCache) automationcheck.Executor {
		var mu sync.Mutex
		return func(ctx context.Context, invocation automationcheck.Invocation) (automationcheck.Evidence, error) {
			mu.Lock()
			defer mu.Unlock()
			evidence, _, err := cache.RunCached(ctx, invocation, input)
			return evidence, err
		}
	}

	t.Run("deterministic composition and cross-recipe reuse", func(t *testing.T) {
		cache := automationcheck.NewEvidenceCache(environment)
		runs := map[string]int{}
		execute := newExecutor(&cache)

		var orderMu sync.Mutex
		var order []string
		observed := func(ctx context.Context, invocation automationcheck.Invocation) (automationcheck.Evidence, error) {
			orderMu.Lock()
			order = append(order, invocation.Check.Name)
			orderMu.Unlock()
			return execute(ctx, invocation)
		}
		first := compose(t, strings.Repeat("b", 64), runs, "")
		results, err := automationcheck.ExecuteDAG(t.Context(), first, nil, observed)
		if err != nil {
			t.Fatal(err)
		}
		if slices.Index(order, "contract") > slices.Index(order, "static") || slices.Index(order, "static") > slices.Index(order, "test") {
			t.Fatalf("composition lost dependency order: %v", order)
		}
		for _, name := range []string{"contract", "static", "test", "lane"} {
			if runs[name] != 1 {
				t.Fatalf("obligation %s ran %d times, want once", name, runs[name])
			}
		}
		if len(results) != 4 {
			t.Fatalf("composition ran %d obligations, want 4", len(results))
		}

		// The same obligations recomposed under a different candidate tree carry
		// different bound invocation ids; every node reuses its recorded evidence.
		second := compose(t, strings.Repeat("c", 64), runs, "")
		if second[0].ID == first[0].ID {
			t.Fatal("distinct compositions share a bound invocation identity")
		}
		reuseResults, err := automationcheck.ExecuteDAG(t.Context(), second, nil, execute)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"contract", "static", "test", "lane"} {
			if runs[name] != 1 {
				t.Fatalf("recomposition re-ran %s: %d executions", name, runs[name])
			}
		}
		for _, result := range reuseResults {
			if !result.Evidence.Reused {
				t.Fatalf("recomposed obligation %s did not reuse its node", result.Invocation.Check.Name)
			}
		}
	})

	t.Run("failure isolates its dependents", func(t *testing.T) {
		cache := automationcheck.NewEvidenceCache(environment)
		runs := map[string]int{}
		execute := newExecutor(&cache)
		bound := compose(t, strings.Repeat("d", 64), runs, "static")
		results, err := automationcheck.ExecuteDAGIndependent(t.Context(), bound, nil, execute)
		if err != nil {
			t.Fatal(err)
		}
		ran := func(name string) bool {
			index := slices.IndexFunc(bound, func(invocation automationcheck.Invocation) bool { return invocation.Check.Name == name })
			return results[index].Err != nil || results[index].Evidence.ID.Valid()
		}
		if !ran("contract") || !ran("static") || ran("test") || !ran("lane") {
			t.Fatalf("failure isolation ran contract=%v static=%v test=%v lane=%v", ran("contract"), ran("static"), ran("test"), ran("lane"))
		}
	})

	t.Run("dependency cycle refused", func(t *testing.T) {
		cycle := []automationcheck.Invocation{
			{ID: artifactRecipeID(t, "left"), Check: automationcheck.Descriptor{Name: "left", Phase: runrecord.PhaseTest, Always: true, Dependencies: []string{"right"}}},
			{ID: artifactRecipeID(t, "right"), Check: automationcheck.Descriptor{Name: "right", Phase: runrecord.PhaseTest, Always: true, Dependencies: []string{"left"}}},
		}
		if _, err := automationcheck.ExecuteDAG(t.Context(), cycle, nil, func(ctx context.Context, invocation automationcheck.Invocation) (automationcheck.Evidence, error) {
			return automationcheck.Run(ctx, invocation)
		}); err == nil {
			t.Fatal("composition admitted a dependency cycle")
		}
	})
}

func artifactRecipeID(t *testing.T, seed string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte(seed))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
