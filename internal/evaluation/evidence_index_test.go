package evaluation

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestEvaluationExecutionDoesNotInventMissingAuthorities keeps unknown execution
// facts distinct from a known raw or hosted protocol. The measured value and its
// evidence identity remain visible even when execution authority is absent.
func TestEvaluationExecutionDoesNotInventMissingAuthorities(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"raw", "hosted", "missing-run", "missing-input", "missing-plan", "missing-policy", "missing-environment", "foreign-source", "foreign-recipe", "foreign-dataset", "foreign-environment", "invalid-policy"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			platform, err := runrecord.NewEnvironment(runrecord.Environment{
				Host: "fixture", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "fixture",
			})
			if mode == "hosted" {
				platform, err = runrecord.NewEnvironment(runrecord.Environment{
					Host: "fixture", OS: "windows", Arch: "amd64", Device: "provider", Backend: runrecord.BackendRemote, Driver: "fixture",
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			if mode != "missing-environment" {
				batch, err := platform.Batch("execution/environment")
				if err != nil {
					t.Fatal(err)
				}
				if _, err := store.Commit(t.Context(), batch); err != nil {
					t.Fatal(err)
				}
			}
			exact, err := CompileExact(exactFixture())
			if err != nil {
				t.Fatal(err)
			}
			authorities := ListingAuthorities()
			authorities.Environment, authorities.CodeCommit = platform.ID, planTestCommit
			if mode == "hosted" {
				authorities.Execution.Prompting = PromptingHostedChat
			}
			plan, err := BindExact(exact, authorities)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "invalid-policy" {
				policy, err := authorityContent(executionContract, ExecutionPolicy{Lifecycle: "automatic"})
				if err != nil {
					t.Fatal(err)
				}
				for index := range plan.authorities {
					if plan.authorities[index].Descriptor.ID == plan.Execution() {
						plan.authorities[index] = policy
					}
				}
				plan.body.Execution = policy.Descriptor.ID
				plan.identity, err = artifact.JSONID(artifact.KindProfile, plan.body)
				if err != nil {
					t.Fatal(err)
				}
			}
			publishPlanFixtureAuthorities(t, store, plan)
			if mode == "missing-policy" {
				plan.authorities = slices.DeleteFunc(plan.authorities, func(content artifact.Content) bool { return content.Descriptor.ID == plan.Execution() })
			}
			output := planID(t, artifact.KindEvaluation, "execution report")
			if _, err := store.Commit(t.Context(), artifact.Batch{Key: "execution/inputs", Artifacts: []artifact.Descriptor{{ID: plan.Identity()}, {ID: plan.Dataset()}, {ID: plan.Execution()}, {ID: output}}}); err != nil {
				t.Fatal(err)
			}
			if mode != "missing-plan" {
				if err := publishAuthorities(t.Context(), store, exact, plan); err != nil {
					t.Fatal(err)
				}
			}
			inputs, commit := []artifact.ID{plan.Identity()}, planTestCommit
			runRecipe, recordDataset, runEnvironment := plan.body.RuntimeRecipe, plan.Dataset(), platform.ID
			if mode == "missing-input" {
				inputs = nil
			}
			if mode == "foreign-source" {
				commit = strings.Repeat("b", len(planTestCommit))
			}
			if mode == "foreign-recipe" {
				runRecipe = planID(t, artifact.KindRecipe, "other recipe")
			}
			if mode == "foreign-dataset" {
				recordDataset = planID(t, artifact.KindDataset, "other dataset")
			}
			if mode == "foreign-environment" {
				runEnvironment = planID(t, artifact.KindEvidence, "other environment")
			}
			if mode == "foreign-recipe" || mode == "foreign-dataset" || mode == "foreign-environment" {
				if _, err := store.Commit(t.Context(), artifact.Batch{Key: "execution/run-parents", Artifacts: []artifact.Descriptor{{ID: runRecipe}, {ID: recordDataset}, {ID: runEnvironment}}}); err != nil {
					t.Fatal(err)
				}
			}
			run, err := runrecord.NewBoundRun(runRecipe, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output}, "", commit, runEnvironment, 1, nil)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "missing-run" {
				if _, err := store.Commit(t.Context(), artifact.Batch{Key: "execution/run", Artifacts: []artifact.Descriptor{{ID: run.ID}}}); err != nil {
					t.Fatal(err)
				}
			} else {
				content, err := run.Content()
				if err != nil {
					t.Fatal(err)
				}
				batch, err := artifact.NewDocumentBatch("execution/run", []artifact.Content{content}, run.Lineage(), nil)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := store.Commit(t.Context(), batch); err != nil {
					t.Fatal(err)
				}
			}
			record, err := runrecord.NewEvaluation(plan.body.RuntimeRecipe, run.ID, recordDataset, []runrecord.Metric{{Name: "accuracy", Value: 0.5, Direction: runrecord.DirectionMaximize}})
			if err != nil {
				t.Fatal(err)
			}
			batch, err := record.Batch("execution/evaluation")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(t.Context(), batch); err != nil {
				t.Fatal(err)
			}
			summaries := LatestEvidence(t.Context(), store, map[artifact.ID]string{recordDataset: "fixture"}, 0).EvaluationsByRecipe[plan.body.RuntimeRecipe]
			if len(summaries) != 1 {
				t.Fatalf("summaries = %+v, want the recorded measurement", summaries)
			}
			summary := summaries[0]
			data, err := json.Marshal(summary)
			var execution struct {
				Known bool `json:"execution_known"`
			}
			if err == nil {
				err = json.Unmarshal(data, &execution)
			}
			known := mode == "raw" || mode == "hosted"
			prompting := "unknown"
			if known {
				prompting = authorities.Execution.Prompting.Label()
			}
			if err != nil || execution.Known != known || summary.Reproducible != (mode == "raw") || summary.Prompting != prompting || summary.Record != record.ID || summary.Metrics["accuracy"] != 0.5 {
				t.Fatalf("summary = %+v (known=%t), want known=%t protocol=%s reproducible=%t with original measurement: %v", summary, execution.Known, known, prompting, mode == "raw", err)
			}
		})
	}
}
