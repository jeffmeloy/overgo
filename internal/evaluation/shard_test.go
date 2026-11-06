package evaluation

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

const shardCaseCount = 5

func TestShardReportsMergeCanonicallyWithoutRetainedCampaign(t *testing.T) {
	t.Parallel()
	exact := testcheck.Must(CompileExact(exactFixture()))(t)
	plan := testcheck.Must(BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleIsolated},
	}))(t)
	cases := make([]artifact.ID, shardCaseCount)
	for index := range cases {
		cases[index] = planID(t, artifact.KindDatasetShard, fmt.Sprintf("case-%d", index))
	}
	shards := testcheck.Must(compileCaseShards(plan, cases))(t)
	if len(shards) != len(cases) || len(shards[0].Cases) != 1 || shards[0].Cases[0] != cases[0] {
		t.Fatalf("shards = %+v", shards)
	}

	reports := make([]shardReport, len(shards))
	var passedCount uint64
	for shardIndex, shard := range shards {
		observations := make([]caseObservation, len(shard.Cases))
		metric := metricState{Name: "exact-accuracy"}
		for caseIndex, caseID := range shard.Cases {
			text := "raw-" + caseID.String()
			output := testcheck.Must(newTextOutput(plan.Identity(), caseID, text))(t)
			passed := (shardIndex+caseIndex)%2 == 0
			observations[caseIndex] = caseObservation{Case: caseID, Output: output.ID, Passed: passed}
			value := float64(0)
			if passed {
				value = 1
				passedCount++
			}
			testcheck.NoError(t, metric.observe(value))
		}
		reports[shardIndex] = testcheck.Must(newShardReport(shard, observations, []metricState{metric}))(t)
	}

	reversed := slices.Clone(reports)
	slices.Reverse(reversed)
	first := testcheck.Must(mergeShardReports(reports))(t)
	second := testcheck.Must(mergeShardReports(reversed))(t)
	if first.ID != second.ID || first.Cases != uint64(len(cases)) || len(first.Metrics) != 1 ||
		first.Metrics[0].Count != uint64(len(cases)) || first.Metrics[0].Sum != float64(passedCount) {
		t.Fatalf("merged report = %+v / %+v", first, second)
	}
	for _, report := range reports {
		for _, observation := range report.Observations {
			if observation.Output.Kind() != artifact.KindOutput {
				t.Fatal("report retained output instead of its identity")
			}
		}
	}
}
