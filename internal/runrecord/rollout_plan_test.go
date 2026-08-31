package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestRolloutPlanIdentityAndValidation pins the rollout plan contract: the
// content-addressed plan binds baseline, candidate, admission evidence,
// cohort key, registered hash version, evidence-derived cohort share,
// observation contract, and typed rollback authority; identical bindings
// reproduce one identity, the parsed document proves it, lineage cites
// every authority exactly once, and a plan missing any binding — or citing
// an unregistered hash version or a hand-picked bound — refuses.
func TestRolloutPlanIdentityAndValidation(t *testing.T) {
	template := RolloutPlan{
		Baseline:       testutil.ArtifactID(t, artifact.KindRecipe, "rollout-baseline"),
		Candidate:      testutil.ArtifactID(t, artifact.KindRecipe, "rollout-candidate"),
		Admission:      testutil.ArtifactID(t, artifact.KindEvidence, "rollout-counterfactual"),
		CohortKey:      "request-digest",
		HashVersion:    RolloutHashVersion,
		CohortShare:    250,
		CohortEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "rollout-cohort-derivation"),
		Observation: RolloutObservation{
			Metric:          "exact-match",
			MinObservations: 128,
			Evidence:        testutil.ArtifactID(t, artifact.KindEvidence, "rollout-observation-derivation"),
		},
		Rollback: testutil.ArtifactID(t, artifact.KindEvidence, "rollout-rollback-decision"),
	}
	plan, err := NewRolloutPlan(template)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.ID.Valid() || plan.ID.Kind() != artifact.KindProfile {
		t.Fatalf("plan identity = %s", plan.ID)
	}
	identical, err := NewRolloutPlan(template)
	if err != nil || identical.ID != plan.ID {
		t.Fatalf("identical bindings produced (%s, %v)", identical.ID, err)
	}
	batch, err := plan.Batch("rollout/plan/" + plan.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseRolloutPlan(batch.Contents[0].Data)
	if err != nil || parsed.ID != plan.ID || parsed.Candidate != plan.Candidate ||
		parsed.CohortShare != 250 || parsed.Observation.MinObservations != 128 {
		t.Fatalf("parsed plan = (%+v, %v)", parsed, err)
	}
	cited := map[artifact.ID]int{}
	for _, edge := range plan.Lineage() {
		if edge.Child != plan.ID {
			t.Fatalf("lineage child = %s", edge.Child)
		}
		cited[edge.Parent]++
	}
	for _, parent := range []artifact.ID{
		template.Baseline, template.Candidate, template.Admission,
		template.CohortEvidence, template.Observation.Evidence, template.Rollback,
	} {
		if cited[parent] != 1 {
			t.Fatalf("lineage cites %s %d times", parent, cited[parent])
		}
	}

	refusals := []struct {
		name   string
		mutate func(*RolloutPlan)
		want   string
	}{
		{"self-rollout", func(p *RolloutPlan) { p.Candidate = p.Baseline }, "must differ from its baseline"},
		{"missing admission", func(p *RolloutPlan) { p.Admission = artifact.ID{} }, "admission evidence"},
		{"missing cohort key", func(p *RolloutPlan) { p.CohortKey = "" }, "cohort key domain"},
		{"unregistered hash", func(p *RolloutPlan) { p.HashVersion = 2 }, "unregistered rollout hash version"},
		{"empty cohort", func(p *RolloutPlan) { p.CohortShare = 0 }, "positive basis-point bound"},
		{"overfull cohort", func(p *RolloutPlan) { p.CohortShare = 10001 }, "positive basis-point bound"},
		{"hand-picked cohort", func(p *RolloutPlan) { p.CohortEvidence = artifact.ID{} }, "derive from exact evidence"},
		{"missing observation", func(p *RolloutPlan) { p.Observation.MinObservations = 0 }, "observation contract"},
		{"unwitnessed observation", func(p *RolloutPlan) {
			p.Observation.Evidence = artifact.ID{}
		}, "observation contract"},
		{"missing rollback authority", func(p *RolloutPlan) { p.Rollback = artifact.ID{} }, "typed rollback authority"},
	}
	for _, refusal := range refusals {
		mutated := template
		refusal.mutate(&mutated)
		if _, err := NewRolloutPlan(mutated); err == nil || !strings.Contains(err.Error(), refusal.want) {
			t.Fatalf("%s admitted: %v", refusal.name, err)
		}
	}
}
