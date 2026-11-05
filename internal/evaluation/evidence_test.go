package evaluation

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

type evidenceBatchDocument interface {
	Batch(string) (artifact.Batch, error)
}

func TestEvaluationEvidenceBindsPlanShardsAndAuthorities(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	exact, plan := ledgerFixture(t, "evidence environment")
	publishPlanFixtureAuthorities(t, store, plan)
	report, err := EvaluateExactSharded(ctx, store, exactGenerator{pieces: []string{"o", "k"}}, exact, plan, nil)
	testcheck.NoError(t, err)
	metrics := []runrecord.Metric{{Name: exactMetricName, Value: 1, Direction: runrecord.DirectionMaximize}}
	policy, err := newAcceptancePolicy(metrics)
	testcheck.NoError(t, err)
	evaluator, err := NewEvaluator(plan.identity, policy)
	testcheck.NoError(t, err)
	run, err := runrecord.NewBoundRun(
		plan.body.RuntimeRecipe, runrecord.OutcomeSucceeded, []artifact.ID{plan.identity}, []artifact.ID{report}, "",
		plan.body.CodeCommit, plan.body.Environment, 10,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: 10}},
	)
	testcheck.NoError(t, err)
	record, err := runrecord.NewEvaluation(plan.body.RuntimeRecipe, run.ID, plan.body.Dataset, metrics)
	testcheck.NoError(t, err)
	for _, publication := range []struct {
		key      string
		document evidenceBatchDocument
	}{{"run", run}, {"evaluation", record}} {
		batch, err := publication.document.Batch("evaluation/evidence-test/" + publication.key)
		testcheck.NoError(t, err)
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
	}
	evidence, err := PublishEvaluationEvidence(ctx, store, plan, policy, evaluator, report, run, record)
	testcheck.NoError(t, err)
	verified, err := RequireEvaluationEvidence(ctx, store, evidence.ID)
	// The document names its run and its evaluation record, and is read with their phases and metrics.
	if err != nil || verified.ID != evidence.ID || !slices.Equal(verified.Phases, run.Phases) || !slices.Equal(verified.Metrics, record.Metrics) {
		t.Fatalf("verified evidence = %+v err=%v", verified, err)
	}
	forged := evaluationEvidenceCodec.Clone(evidence)
	forged.ID = artifact.ID{}
	forged.CodeCommit = "fedcba9876543210fedcba9876543210fedcba98"
	forged, err = evaluationEvidenceCodec.New(forged)
	testcheck.NoError(t, err)
	forgedBatch, err := evaluationEvidenceCodec.Batch(
		"evaluation/evidence-test/forged-bypass", forged, forged.Lineage(), nil,
	)
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, forgedBatch); err != nil {
		t.Fatal(err)
	}
	rejected, err := RequireEvaluationEvidence(ctx, store, forged.ID)
	if err == nil || rejected.ID.Valid() || rejected.Metrics != nil {
		t.Fatalf("read-side verification accepted forged evidence: %+v err=%v", rejected, err)
	}
	stored, found, err := evaluationEvidenceCodec.Read(ctx, store, evidence.ID)
	if err != nil || !found || stored.Plan != plan.identity || stored.Acceptance != policy.ID ||
		stored.Report != report || stored.Run != run.ID || stored.Evaluation != record.ID ||
		stored.Split != plan.body.Split || len(stored.Shards) != len(exact.suite.Cases) ||
		len(stored.Phases) != 0 || len(stored.Metrics) != 0 {
		t.Fatalf("stored=%+v found=%v err=%v", stored, found, err)
	}
	parents, err := store.Parents(ctx, evidence.ID)
	if err != nil || len(parents) < len(stored.Shards)+10 {
		t.Fatalf("evidence lineage=%+v err=%v", parents, err)
	}
	wrong, err := newAcceptancePolicy([]runrecord.Metric{{
		Name: "other", Direction: runrecord.DirectionMaximize,
	}})
	testcheck.NoError(t, err)
	if _, err := PublishEvaluationEvidence(ctx, store, plan, wrong, evaluator, report, run, record); err == nil {
		t.Fatal("accepted evaluation outside suite policy")
	}

	planContent, err := plan.Content()
	testcheck.NoError(t, err)
	evidenceContent, err := evidence.Content()
	testcheck.NoError(t, err)
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
			testcheck.NoError(t, json.Indent(&formatted, fixture.content.Data, "", "  "))
			canonical, err := fixture.normalize(formatted.Bytes())
			if err != nil || !bytes.Equal(canonical, fixture.content.Data) {
				t.Fatalf("normalized %s differs: %v", name, err)
			}
			var foreign map[string]any
			testcheck.NoError(t, json.Unmarshal(fixture.content.Data, &foreign))
			foreign["external_id"] = "foreign-authority"
			injected, err := json.Marshal(foreign)
			testcheck.NoError(t, err)
			if _, err := fixture.normalize(injected); err == nil {
				t.Fatal("foreign authority field accepted")
			}
		})
	}
}

func PublishEvaluationEvidence(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	acceptance AcceptancePolicy,
	evaluator Evaluator,
	report artifact.ID,
	run runrecord.Run,
	record runrecord.Evaluation,
	causal ...runrecord.CausalContext,
) (EvaluationEvidence, error) {
	evidence, err := newEvaluationEvidence(
		ctx, repository, plan, acceptance, evaluator, report, run, record, nil, causal...,
	)
	if err != nil {
		return EvaluationEvidence{}, err
	}
	batch := artifact.Batch{Key: "evaluation/evidence/" + evidence.ID.String()}
	if err := appendEvaluationEvidence(&batch, acceptance, evaluator, evidence); err != nil {
		return EvaluationEvidence{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return evidence, err
}
