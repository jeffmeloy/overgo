package scorecard

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestATimingIsReadUnderOneRuntimePolicy holds a serving timing to the policy its attempts ran under: a sampling policy
// changes without its recipe, so two older attempts that decoded at 100 tokens a second under one policy are no repeats
// of the newest, which decoded at 1 under another. The rate reads the newest's 1 and names its policy, where on
// 2026-11-09 the median of the three read 100 (the owner's review, its numbers inputs and not a measured model). Read
// against the first policy's 100, the 1 is a decline whose verdict names both policies.
func TestATimingIsReadUnderOneRuntimePolicy(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	model := testutil.ArtifactID(t, artifact.KindModel, "policy model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "policy recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "policy environment")
	first, second := testutil.ArtifactID(t, artifact.KindProfile, "policy A"), testutil.ArtifactID(t, artifact.KindProfile, "policy B")
	const tokens, fast, slow = 100, time.Second, 100 * time.Second
	// attempt is one serving attempt, oldest first: the policy it ran under and how long its decode took.
	type attempt struct {
		policy artifact.ID
		decode time.Duration
	}
	// rate is the throughput cell a store holding the attempts reads.
	rate := func(attempts ...attempt) Cell {
		t.Helper()
		store := overgodbtest.Open(t, t.TempDir())
		testcheck.Must(store.Commit(ctx, artifact.Batch{Key: "policy/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipeID}, {ID: environment}}}))(t)
		for index, each := range attempts {
			observation := testcheck.Must(runrecord.NewServingObservation(runrecord.ServingObservation{
				Model: model, Recipe: recipeID, Environment: environment, Task: recipe.TaskInference, Outcome: runrecord.OutcomeSucceeded,
				StartedUnixNS: int64(index + 1), MeasuredNS: uint64(each.decode), Policy: each.policy, Usage: runrecord.ServingUsage{OutputTokens: tokens},
				Phases: []runrecord.PhaseMetric{{Phase: runrecord.PhaseLoad, DurationNS: 1}, {Phase: runrecord.PhaseDecode, DurationNS: uint64(each.decode)}},
			}))(t)
			testcheck.Must(store.Commit(ctx, testcheck.Must(observation.Batch(fmt.Sprintf("policy/%d", index)))(t)))(t)
		}
		return throughputCell(recipe.TaskInference, evaluation.BenchmarkSummary{}, false, testcheck.Must(servingByRecipe(ctx, store))(t)[recipeID])
	}
	then := rate(attempt{first, fast}, attempt{first, fast})
	now := rate(attempt{first, fast}, attempt{first, fast}, attempt{second, slow})
	if now.Value != tokens/slow.Seconds() || now.Policy != second || then.Value != tokens/fast.Seconds() || then.Policy != first {
		t.Fatalf("the rate reads %v tok/s under %s, the first policy's %v under %s; want %v under %s and %v under %s",
			now.Value, now.Policy, then.Value, then.Policy, tokens/slow.Seconds(), second, tokens/fast.Seconds(), first)
	}
	if verdict, why := holds(metric.Throughput, then, now); verdict == holding || !strings.Contains(why, "run under policy "+second.String()) {
		t.Fatalf("the rate's fall from %v to %v across a policy change reads %v: %q, want a decline naming both policies", then.Value, now.Value, verdict, why)
	}
}
