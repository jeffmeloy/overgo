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
	exact, err := CompileExact(exactFixture())
	testcheck.NoError(t, err)
	plan, err := BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleIsolated},
	})
	testcheck.NoError(t, err)
	cases := make([]artifact.ID, shardCaseCount)
	for index := range cases {
		cases[index] = planID(t, artifact.KindDatasetShard, fmt.Sprintf("case-%d", index))
	}
	shards, err := compileCaseShards(plan, cases)
	testcheck.NoError(t, err)
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
			output, err := newTextOutput(plan.Identity(), caseID, text)
			testcheck.NoError(t, err)
			passed := (shardIndex+caseIndex)%2 == 0
			observations[caseIndex] = caseObservation{Case: caseID, Output: output.ID, Passed: passed}
			value := float64(0)
			if passed {
				value = 1
				passedCount++
			}
			testcheck.NoError(t, metric.observe(value))
		}
		reports[shardIndex], err = newShardReport(shard, observations, []metricState{metric})
		testcheck.NoError(t, err)
	}

	reversed := slices.Clone(reports)
	slices.Reverse(reversed)
	first, err := mergeShardReports(reports)
	testcheck.NoError(t, err)
	second, err := mergeShardReports(reversed)
	testcheck.NoError(t, err)
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
