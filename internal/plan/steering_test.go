package plan

import (
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"testing"
)

func TestSteeringProposalAdmission(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	evaluation := testutil.ArtifactID(t, artifact.KindProfile, "evaluation")
	measurement := testutil.ArtifactID(t, artifact.KindEvidence, "measurement")
	affected := testutil.ArtifactID(t, artifact.KindProfile, "policy")
	verifier := testutil.ArtifactID(t, artifact.KindRecipe, "verifier")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "steering-authorities", Artifacts: []artifact.Descriptor{{ID: evaluation}, {ID: measurement}, {ID: affected}, {ID: verifier}}}); err != nil {
		t.Fatal(err)
	}
	proposal, err := recipe.NewSteeringProposal(recipe.SteeringProposal{Goal: "Reduce verification cost without coverage loss.", Prediction: recipe.SteeringPrediction{Metric: "gate wall time", Benefit: 0.2, Cost: 10, Unit: "ratio", Uncertainty: 0.1}, Evaluation: evaluation, AffectedAuthorities: []artifact.ID{affected}, Measurements: []artifact.ID{measurement}, Rows: []recipe.SteeringPlanRow{{Item: "gate-efficiency", Step: "measure", Title: "Measure structural exclusions", Verifier: verifier, Capabilities: []string{"gate"}}}})
	if err != nil {
		t.Fatal(err)
	}
	current := Plan{Campaign: "future", Doctrine: "measured", Items: []Item{}}
	admitted, err := AdmitSteeringProposal(ctx, store, current, proposal, map[artifact.ID]string{verifier: "go test ./cmd/gate"})
	if err != nil {
		t.Fatal(err)
	}
	if len(admitted.Items) != 1 || admitted.Items[0].Steps[0].Verify != "go test ./cmd/gate" {
		t.Fatalf("admitted = %+v", admitted)
	}
	if _, err := AdmitSteeringProposal(ctx, store, current, proposal, nil); err == nil {
		t.Fatal("unbound model verifier was admitted")
	}
}
