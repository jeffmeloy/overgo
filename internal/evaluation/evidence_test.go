package evaluation

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

type evidenceBatchDocument interface {
	Batch(string) (artifact.Batch, error)
}

func TestEvaluationEvidenceBindsPlanShardsAndAuthorities(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	exact, plan := ledgerFixture(t, "evidence environment")
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := EvaluateExactSharded(ctx, store, exactGenerator{pieces: []string{"o", "k"}}, exact, plan, nil)
	if err != nil {
		t.Fatal(err)
	}
	metrics := []runrecord.Metric{{Name: exactMetricName, Value: 1, Direction: runrecord.DirectionMaximize}}
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
		batch, err := publication.document.Batch("evaluation/evidence-test/" + publication.key)
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
	stored, found, err := evaluationEvidenceCodec.Read(ctx, store, evidence.ID)
	if err != nil || !found || stored.Plan != plan.identity || stored.Acceptance != policy.ID ||
		stored.Report != report || stored.Run != run.ID || stored.Evaluation != record.ID ||
		stored.Split != plan.body.Split || len(stored.Shards) != len(exact.suite.Cases) ||
		len(stored.Phases) != 1 || len(stored.Metrics) != 1 {
		t.Fatalf("stored=%+v found=%v err=%v", stored, found, err)
	}
	parents, err := store.Parents(ctx, evidence.ID)
	if err != nil || len(parents) < len(stored.Shards)+10 {
		t.Fatalf("evidence lineage=%+v err=%v", parents, err)
	}
	wrong, err := newAcceptancePolicy([]runrecord.Metric{{
		Name: "other", Direction: runrecord.DirectionMaximize,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishEvaluationEvidence(ctx, store, plan, wrong, evaluator, report, run, record); err == nil {
		t.Fatal("accepted evaluation outside suite policy")
	}

	planContent, err := plan.Content()
	if err != nil {
		t.Fatal(err)
	}
	evidenceContent, err := evidence.Content()
	if err != nil {
		t.Fatal(err)
	}
	for name, fixture := range map[string]struct {
		content   artifact.Content
		normalize func([]byte) ([]byte, error)
	}{
		"plan": {content: planContent, normalize: func(data []byte) ([]byte, error) {
			_, canonical, normalizeErr := NormalizeEvaluationPlan(data)
			return canonical, normalizeErr
		}},
		"evidence": {content: evidenceContent, normalize: func(data []byte) ([]byte, error) {
			_, canonical, normalizeErr := NormalizeEvaluationEvidence(data)
			return canonical, normalizeErr
		}},
	} {
		t.Run("external "+name, func(t *testing.T) {
			var formatted bytes.Buffer
			if err := json.Indent(&formatted, fixture.content.Data, "", "  "); err != nil {
				t.Fatal(err)
			}
			canonical, err := fixture.normalize(formatted.Bytes())
			if err != nil || !bytes.Equal(canonical, fixture.content.Data) {
				t.Fatalf("normalized %s differs: %v", name, err)
			}
			var foreign map[string]any
			if err := json.Unmarshal(fixture.content.Data, &foreign); err != nil {
				t.Fatal(err)
			}
			foreign["external_id"] = "foreign-authority"
			injected, err := json.Marshal(foreign)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := fixture.normalize(injected); err == nil {
				t.Fatal("foreign authority field accepted")
			}
		})
	}
}
