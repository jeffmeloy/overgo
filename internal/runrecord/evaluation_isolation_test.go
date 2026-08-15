package runrecord

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
)

func TestSplitIsolationAndQueryBudget(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	owner := id(artifact.KindEvidence, "evaluation-authority")
	dataset := id(artifact.KindDataset, "held-out-dataset")
	expires := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	type surface struct {
		name  string
		split artifact.ID
		grant BudgetGrant
	}
	surfaces := make([]surface, 4)
	for index, name := range []string{"development", "selection", "promotion", "audit"} {
		split := id(artifact.KindDatasetShard, name)
		grant, err := NewBudgetGrant(BudgetGrant{
			Owner: owner, Subject: split, Unit: "query", Issued: uint64(index + 1), ExpiresAt: expires.Format(time.RFC3339Nano),
		})
		if err != nil {
			t.Fatal(err)
		}
		surfaces[index] = surface{name: name, split: split, grant: grant}
	}
	policy, err := trainingdata.NewEvaluationIsolation(trainingdata.EvaluationIsolation{
		Dataset:     dataset,
		Development: trainingdata.SplitBinding{Split: surfaces[0].split, Budget: surfaces[0].grant.ID},
		Selection:   trainingdata.SplitBinding{Split: surfaces[1].split, Budget: surfaces[1].grant.ID},
		Promotion:   trainingdata.SplitBinding{Split: surfaces[2].split, Budget: surfaces[2].grant.ID},
		Audit:       trainingdata.SplitBinding{Split: surfaces[3].split, Budget: surfaces[3].grant.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	for index, surface := range surfaces {
		budget, err := policy.BudgetFor(surface.split)
		if err != nil || budget != surface.grant.ID || surface.grant.Subject != surface.split {
			t.Fatalf("%s budget binding = (%s, %v)", surface.name, budget, err)
		}
		if allowed := policy.AuthorizeProposer(surface.split) == nil; allowed != (index < 2) {
			t.Fatalf("proposer visibility for %s = %v", surface.name, allowed)
		}
		if _, err := surface.grant.Batch("fixture/budget/" + surface.name); err != nil {
			t.Fatal(err)
		}
	}
	if err := surfaces[1].grant.Allows(1, 1, expires.Add(-time.Second)); err != nil {
		t.Fatal(err)
	}
	if err := surfaces[1].grant.Allows(2, 1, expires.Add(-time.Second)); err == nil {
		t.Fatal("exhausted query budget accepted")
	}
	if err := surfaces[3].grant.Allows(0, 1, expires); err == nil {
		t.Fatal("expired query budget accepted")
	}
	if _, err := policy.Batch("fixture/evaluation-isolation"); err != nil {
		t.Fatal(err)
	}
	invalid := policy
	invalid.Audit = invalid.Promotion
	if _, err := trainingdata.NewEvaluationIsolation(invalid); err == nil {
		t.Fatal("shared promotion and audit split accepted")
	}
}
