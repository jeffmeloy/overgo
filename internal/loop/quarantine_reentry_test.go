package loop

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func publishReentryEvidence(t *testing.T, store artifact.Repository, name string) artifact.ID {
	t.Helper()
	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/json", Schema: "overgo/test-evidence/v1",
	}
	content, err := artifact.JSONContent(contract, map[string]string{"fixture": name})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"live-safety/reentry-fixture/evidence/"+name, []artifact.Content{content}, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	return content.Descriptor.ID
}

// TestQuarantineReentryRequiresNewEvidence pins the reentry door: a
// quarantined candidate re-enters rollout only with the prior finding
// named, changed-mechanism and evaluation evidence introduced after that
// finding — pre-quarantine evidence replayed through a restart proves
// nothing — a fresh rollout plan governing exactly the candidate against
// the restored baseline, and an explicit recorded decider. Every gap
// refuses by name, and the admitted record cites the whole closure.
func TestQuarantineReentryRequiresNewEvidence(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "promotion-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "live-safety/reentry-fixture/facts",
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: 4096},
			{ID: testutil.ArtifactID(t, artifact.KindProfile, "promotion-profile")},
			{ID: testutil.ArtifactID(t, artifact.KindModelDefinition, "promotion-definition")},
		},
	}); err != nil {
		t.Fatal(err)
	}
	staleEvidence := publishReentryEvidence(t, store, "stale-evaluation")
	baseline := promotionDefinition(t, modelID, recipe.ResidencyHostReference)
	candidate := promotionDefinition(t, modelID, recipe.ResidencyHostCache)
	for name, definition := range map[string]recipe.Definition{
		"baseline": baseline, "candidate": candidate,
	} {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "live-safety/reentry-fixture/"+name, definition,
		); err != nil {
			t.Fatal(err)
		}
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, baseline,
		promotionVerification(t, store, baseline.ID, "live-safety/reentry-fixture/baseline-verification"),
		recipe.EvidenceVerified, "baseline in service",
	); err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, candidate,
		promotionVerification(t, store, candidate.ID, "live-safety/reentry-fixture/candidate-verification"),
		recipe.EvidenceVerified, "candidate promoted",
	); err != nil {
		t.Fatal(err)
	}
	finding, err := RollbackQuarantinedCandidate(
		ctx, store, modelID, recipe.TaskInference, candidate.ID,
		publishBreakerTransition(t, store, 2),
		promotionFailedVerification(t, store, candidate.ID, "live-safety/reentry-fixture/failed-verification"),
		promotionVerification(t, store, baseline.ID, "live-safety/reentry-fixture/reactivation-verification"),
		"sustained live regression",
	)
	if err != nil {
		t.Fatal(err)
	}

	mechanism := publishReentryEvidence(t, store, "changed-mechanism")
	evaluation := publishReentryEvidence(t, store, "new-evaluation")
	planAuthorities := []artifact.ID{
		testutil.ArtifactID(t, artifact.KindEvidence, "reentry-counterfactual"),
		testutil.ArtifactID(t, artifact.KindEvidence, "reentry-cohort-derivation"),
		testutil.ArtifactID(t, artifact.KindEvidence, "reentry-observation-derivation"),
		testutil.ArtifactID(t, artifact.KindEvidence, "reentry-rollback-decision"),
	}
	descriptors := make([]artifact.Descriptor, 0, len(planAuthorities))
	for _, id := range planAuthorities {
		descriptors = append(descriptors, artifact.Descriptor{ID: id})
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "live-safety/reentry-fixture/new-evidence", Artifacts: descriptors,
	}); err != nil {
		t.Fatal(err)
	}
	freshPlan, err := runrecord.NewRolloutPlan(runrecord.RolloutPlan{
		Baseline: baseline.ID, Candidate: candidate.ID,
		Admission: planAuthorities[0], CohortKey: "request-digest",
		HashVersion: runrecord.RolloutHashVersion, CohortShare: 100,
		CohortEvidence: planAuthorities[1],
		Observation: runrecord.RolloutObservation{
			Metric: "exact-match", MinObservations: 64, Evidence: planAuthorities[2],
		},
		Rollback: planAuthorities[3],
	})
	if err != nil {
		t.Fatal(err)
	}
	planBatch, err := freshPlan.Batch("live-safety/reentry-fixture/fresh-plan")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, planBatch); err != nil {
		t.Fatal(err)
	}

	complete := QuarantineReentry{
		Candidate: candidate.ID, Finding: finding, Mechanism: mechanism,
		Evaluation: evaluation, Plan: freshPlan.ID,
		Decider: "operator:jeffm", Reason: "attention mechanism replaced and re-evaluated",
	}
	refusals := []struct {
		name   string
		mutate func(*QuarantineReentry)
		want   string
	}{
		{"unsigned", func(r *QuarantineReentry) { r.Decider = " " }, "explicit recorded decider"},
		{"unnamed finding", func(r *QuarantineReentry) { r.Finding = mechanism }, "published rollback finding"},
		{"replayed evaluation", func(r *QuarantineReentry) { r.Evaluation = staleEvidence }, "predates the quarantine finding"},
		{"replayed mechanism", func(r *QuarantineReentry) { r.Mechanism = staleEvidence }, "predates the quarantine finding"},
		{"foreign plan", func(r *QuarantineReentry) { r.Plan = staleEvidence }, "predates the quarantine finding"},
	}
	for _, refusal := range refusals {
		mutated := complete
		refusal.mutate(&mutated)
		if _, err := AdmitQuarantineReentry(ctx, store, mutated); err == nil ||
			!strings.Contains(err.Error(), refusal.want) {
			t.Fatalf("%s reentry admitted: %v", refusal.name, err)
		}
	}

	admitted, err := AdmitQuarantineReentry(ctx, store, complete)
	if err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(ctx, admitted)
	if err != nil {
		t.Fatal(err)
	}
	cited := map[artifact.ID]bool{}
	for _, edge := range parents {
		cited[edge.Parent] = true
	}
	for _, required := range []artifact.ID{candidate.ID, finding, mechanism, evaluation, freshPlan.ID} {
		if !cited[required] {
			t.Fatalf("reentry record does not cite %s: %v", required, parents)
		}
	}
}
