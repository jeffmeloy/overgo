package evaluation

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// publishSelectionEvidence publishes one complete evaluation evidence chain
// — committed recipe definition, plan authorities, sharded report, run,
// record, evidence — for a model of the given resource size, measuring the
// exact fixture metric at the given quality.
func publishSelectionEvidence(
	t *testing.T, store artifact.Repository, name string, resource uint64, quality float64,
) (artifact.ID, recipe.Definition) {
	t.Helper()
	ctx := context.Background()
	modelID := planID(t, artifact.KindModel, name+"-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "routing/fixture/model/" + name,
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: resource},
			{ID: planID(t, artifact.KindProfile, name+"-profile")},
			{ID: planID(t, artifact.KindModelDefinition, name+"-definition")},
			{ID: planID(t, artifact.KindEvidence, name+"-environment")},
		},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		modelID, planID(t, artifact.KindProfile, name+"-profile"),
		planID(t, artifact.KindModelDefinition, name+"-definition"),
		recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostReference,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "routing/fixture/candidate/"+name, definition,
	); err != nil {
		t.Fatal(err)
	}
	exact, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	plan, err := BindExact(exact, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, name+"-definition"),
		RuntimeRecipe:   definition.ID,
		CodeCommit:      planTestCommit,
		Environment:     planID(t, artifact.KindEvidence, name+"-environment"),
		Execution:       ExecutionPolicy{Lifecycle: LifecycleIsolated},
	})
	if err != nil {
		t.Fatal(err)
	}
	report, err := EvaluateExactSharded(ctx, store, exactGenerator{pieces: []string{"o", "k"}}, exact, plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	metrics := []runrecord.Metric{{Name: exactMetricName, Value: quality, Direction: runrecord.DirectionMaximize}}
	policy, err := newAcceptancePolicy(metrics)
	if err != nil {
		t.Fatal(err)
	}
	evaluator, err := NewEvaluator(plan.identity, policy)
	if err != nil {
		t.Fatal(err)
	}
	run, err := runrecord.NewBoundRun(
		plan.body.RuntimeRecipe, runrecord.OutcomeSucceeded, []artifact.ID{plan.identity}, []artifact.ID{report}, "",
		plan.body.CodeCommit, plan.body.Environment, 10,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: 10}},
	)
	if err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewEvaluation(plan.body.RuntimeRecipe, run.ID, plan.body.Dataset, metrics)
	if err != nil {
		t.Fatal(err)
	}
	for _, publication := range []struct {
		key      string
		document evidenceBatchDocument
	}{{"run", run}, {"evaluation", record}} {
		batch, err := publication.document.Batch("routing/fixture/" + name + "/" + publication.key)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	evidence, err := PublishEvaluationEvidence(ctx, store, plan, policy, evaluator, report, run, record)
	if err != nil {
		t.Fatal(err)
	}
	return evidence.ID, definition
}

// TestEvidenceDerivedSelection pins the live selection contract: candidates
// admit only through verified published evaluation evidence on the
// baseline's split measuring the baseline's metric, quality and threshold
// derive from those measurements, resource derives from the published model
// artifact size, the cheapest candidate meeting the baseline is selected,
// and the decision record commits to the store before the selection is
// returned — the benchmark substrate acting as a control input, with every
// decision on the record.
func TestEvidenceDerivedSelection(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	baseline, _ := publishSelectionEvidence(t, store, "incumbent", 8192, 0.5)
	cheapStrong, cheapDefinition := publishSelectionEvidence(t, store, "cheap-strong", 2048, 0.6)
	costlyStrong, _ := publishSelectionEvidence(t, store, "costly-strong", 16384, 0.9)
	cheapestWeak, _ := publishSelectionEvidence(t, store, "cheapest-weak", 1024, 0.3)

	decision, err := SelectServingRecipe(
		ctx, store, recipe.TaskInference, exactMetricName, baseline,
		[]artifact.ID{costlyStrong, cheapStrong, cheapestWeak},
	)
	if err != nil {
		t.Fatal(err)
	}
	if decision.Selected != cheapDefinition.ID {
		t.Fatalf("selection = %s; the cheapest candidate meeting the baseline must win", decision.Selected)
	}
	if decision.Signal.QualityThreshold != 0.5 || decision.Signal.ThresholdEvidence != baseline ||
		decision.Signal.Capability != exactMetricName || len(decision.Candidates) != 3 {
		t.Fatalf("decision signal = %+v candidates=%d", decision.Signal, len(decision.Candidates))
	}
	content, found, err := artifact.ReadContent(ctx, store, decision.ID)
	if err != nil || !found {
		t.Fatalf("decision was not recorded: (%v, %v)", found, err)
	}
	recorded, err := modelrecipe.ParseRoutingDecision(content.Data)
	if err != nil || recorded.ID != decision.ID || recorded.Selected != decision.Selected {
		t.Fatalf("recorded decision = (%s, %v)", recorded.ID, err)
	}

	if _, err := SelectServingRecipe(
		ctx, store, recipe.TaskInference, exactMetricName, costlyStrong,
		[]artifact.ID{cheapestWeak},
	); err == nil || !strings.Contains(err.Error(), "meets the routing threshold") {
		t.Fatalf("no eligible candidate still selected: %v", err)
	}
	if _, err := SelectServingRecipe(
		ctx, store, recipe.TaskInference, "unmeasured-metric", baseline,
		[]artifact.ID{cheapStrong},
	); err == nil || !strings.Contains(err.Error(), "does not measure") {
		t.Fatalf("unmeasured metric selected: %v", err)
	}
	forged := planID(t, artifact.KindEvidence, "never-published")
	if _, err := SelectServingRecipe(
		ctx, store, recipe.TaskInference, exactMetricName, baseline,
		[]artifact.ID{forged},
	); err == nil {
		t.Fatal("unadmitted candidate evidence selected")
	}
}
