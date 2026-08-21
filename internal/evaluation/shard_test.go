package evaluation

import (
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
)

const shardCaseCount = 5

func TestShardReportsMergeCanonicallyWithoutRetainedCampaign(t *testing.T) {
	exact, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
		RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, "environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleIsolated},
	})
	if err != nil {
		t.Fatal(err)
	}
	cases := make([]artifact.ID, shardCaseCount)
	for index := range cases {
		cases[index] = planID(t, artifact.KindDatasetShard, fmt.Sprintf("case-%d", index))
	}
	shards, err := compileCaseShards(plan, cases)
	if err != nil {
		t.Fatal(err)
	}
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
			if err != nil {
				t.Fatal(err)
			}
			passed := (shardIndex+caseIndex)%2 == 0
			observations[caseIndex] = caseObservation{Case: caseID, Output: output.ID, Passed: passed}
			value := float64(0)
			if passed {
				value = 1
				passedCount++
			}
			if err := metric.observe(value); err != nil {
				t.Fatal(err)
			}
		}
		reports[shardIndex], err = newShardReport(shard, observations, []metricState{metric})
		if err != nil {
			t.Fatal(err)
		}
	}

	reversed := slices.Clone(reports)
	slices.Reverse(reversed)
	first, err := mergeShardReports(reports)
	if err != nil {
		t.Fatal(err)
	}
	second, err := mergeShardReports(reversed)
	if err != nil {
		t.Fatal(err)
	}
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
