package evaluation

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func projectionRolloutPlan(t *testing.T, store artifact.Repository) runrecord.RolloutPlan {
	t.Helper()
	ctx := t.Context()
	template := runrecord.RolloutPlan{
		Baseline:       planID(t, artifact.KindRecipe, "projection-baseline"),
		Candidate:      planID(t, artifact.KindRecipe, "projection-candidate"),
		Admission:      planID(t, artifact.KindEvidence, "projection-counterfactual"),
		CohortKey:      "request-digest",
		HashVersion:    runrecord.RolloutHashVersion,
		CohortShare:    500,
		CohortEvidence: planID(t, artifact.KindEvidence, "projection-cohort-derivation"),
		Observation: runrecord.RolloutObservation{
			Metric:          "exact-match",
			MinObservations: 32,
			Evidence:        planID(t, artifact.KindEvidence, "projection-observation-derivation"),
		},
		Rollback: planID(t, artifact.KindEvidence, "projection-rollback-decision"),
	}
	authorities := []artifact.Descriptor{
		{ID: template.Baseline}, {ID: template.Candidate}, {ID: template.Admission},
		{ID: template.CohortEvidence}, {ID: template.Observation.Evidence}, {ID: template.Rollback},
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "rollout/projection-fixture/authorities", Artifacts: authorities,
	}); err != nil {
		t.Fatal(err)
	}
	plan, err := runrecord.NewRolloutPlan(template)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := plan.Batch("rollout/projection-fixture/plan")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	return plan
}

func commitRolloutObservation(t *testing.T, store artifact.Repository, plan artifact.ID, name string) artifact.ID {
	t.Helper()
	observation := planID(t, artifact.KindEvidence, name)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "rollout/projection-fixture/" + name,
		Artifacts: []artifact.Descriptor{{ID: observation}},
		Lineage:   []artifact.Lineage{{Child: observation, Parent: plan, Relation: artifact.RelationDependsOn}},
	}); err != nil {
		t.Fatal(err)
	}
	return observation
}

// TestRolloutProjectionEvidenceReproduces pins the projection evidence
// contract: a reading binds head, projector version, query contract,
// consumed sources, coverage, and result digest; projecting twice at one
// head reproduces one identity bit for bit, a reading at a new head is new
// evidence, prior projection reports never feed later readings, a tampered
// digest refuses on parse, and promotion never consumes a mutable row.
func TestRolloutProjectionEvidenceReproduces(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	plan := projectionRolloutPlan(t, store)
	first := commitRolloutObservation(t, store, plan.ID, "observation-first")
	second := commitRolloutObservation(t, store, plan.ID, "observation-second")

	reading, err := ProjectRolloutEvidence(ctx, store, plan.ID)
	if err != nil {
		t.Fatal(err)
	}
	if reading.Coverage != 2 || len(reading.Sources) != 2 ||
		reading.Projector != RolloutProjectorVersion || reading.Query != RolloutProjectionQuery {
		t.Fatalf("projection = %+v", reading)
	}
	cited := map[artifact.ID]bool{}
	for _, source := range reading.Sources {
		cited[source] = true
	}
	if !cited[first] || !cited[second] {
		t.Fatalf("projection lost sources: %v", reading.Sources)
	}
	again, err := ProjectRolloutEvidence(ctx, store, plan.ID)
	if err != nil || again.ID != reading.ID || again.Digest != reading.Digest {
		t.Fatalf("same-head projection drifted: (%s vs %s, %v)", again.Digest, reading.Digest, err)
	}
	if _, err := PublishRolloutProjection(ctx, store, reading); err != nil {
		t.Fatal(err)
	}
	content, found, err := artifact.ReadContent(ctx, store, reading.ID)
	if err != nil || !found {
		t.Fatalf("projection was not published: (%v, %v)", found, err)
	}
	published, err := ParseRolloutProjection(content.Data)
	if err != nil || published.ID != reading.ID || published.Digest != reading.Digest {
		t.Fatalf("published projection = (%+v, %v)", published, err)
	}

	// A reading at a new head is new evidence, and the prior projection
	// report is never a source of the next reading.
	third := commitRolloutObservation(t, store, plan.ID, "observation-third")
	moved, err := ProjectRolloutEvidence(ctx, store, plan.ID)
	if err != nil {
		t.Fatal(err)
	}
	if moved.ID == reading.ID || moved.Digest == reading.Digest || moved.Coverage != 3 ||
		moved.Sequence <= reading.Sequence {
		t.Fatalf("new-head projection = %+v", moved)
	}
	for _, source := range moved.Sources {
		if source == reading.ID {
			t.Fatal("projection consumed a prior projection report")
		}
		if source != first && source != second && source != third {
			t.Fatalf("projection consumed foreign source %s", source)
		}
	}

	var tampered map[string]any
	if err := json.Unmarshal(content.Data, &tampered); err != nil {
		t.Fatal(err)
	}
	tampered["digest"] = strings.Repeat("0", 64)
	forged, err := json.Marshal(tampered)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseRolloutProjection(forged); err == nil ||
		!strings.Contains(err.Error(), "does not reproduce") {
		t.Fatalf("tampered digest parsed: %v", err)
	}

	if _, err := ProjectRolloutEvidence(
		ctx, store, planID(t, artifact.KindProfile, "never-committed-plan"),
	); err == nil || !strings.Contains(err.Error(), "is not committed") {
		t.Fatalf("uncommitted plan projected: %v", err)
	}
	if _, err := ProjectRolloutEvidence(ctx, store, first); err == nil {
		t.Fatal("non-plan artifact projected")
	}
}
