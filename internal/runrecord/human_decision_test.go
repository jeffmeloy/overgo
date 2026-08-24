package runrecord

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestDecisionRejectsArgumentSubstitution(t *testing.T) {
	store, request, decision := humanDecisionFixture(t)
	decision.Arguments[0] = "substituted"
	if err := PublishHumanDecision(context.Background(), store, request, decision); err == nil {
		t.Fatal("substituted decision was published")
	}
}

func TestDecisionRejectsReplay(t *testing.T) {
	store, request, decision := humanDecisionFixture(t)
	if err := PublishHumanDecision(context.Background(), store, request, decision); err != nil {
		t.Fatal(err)
	}
	if err := PublishHumanDecision(context.Background(), store, request, decision); err == nil {
		t.Fatal("decision replay was published")
	}
}

func humanDecisionFixture(t *testing.T) (*overgodb.Store, operatoraction.ApprovalRequest, HumanDecision) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "operation")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	action := operatoraction.Action{Code: "resume", Summary: "Resume operation", Argv: []string{"overgo", "resume"}}
	request, err := operatoraction.NewApprovalRequest(operation, recipeID, action, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := NewHumanDecision(request, operatoraction.AnswerGrant)
	if err != nil {
		t.Fatal(err)
	}
	return store, request, decision
}
