package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestDecisionRejectsArgumentSubstitution(t *testing.T) {
	store, request, decision := humanDecisionFixture(t)
	decision.Arguments[0] = "substituted"
	if err := PublishHumanDecision(t.Context(), store, request, decision); err == nil {
		t.Fatal("substituted decision was published")
	}
}

func TestDecisionRejectsReplay(t *testing.T) {
	store, request, decision := humanDecisionFixture(t)
	testcheck.NoError(t, PublishHumanDecision(t.Context(), store, request, decision))
	if err := PublishHumanDecision(t.Context(), store, request, decision); err == nil {
		t.Fatal("decision replay was published")
	}
}

func humanDecisionFixture(t *testing.T) (*overgodb.Store, operatoraction.ApprovalRequest, HumanDecision) {
	t.Helper()
	store := overgodbtest.Open(t, t.TempDir())
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "operation")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	action := operatoraction.Action{Code: "resume", Summary: "Resume operation", Argv: []string{"overgo", "resume"}}
	request := testcheck.Must(operatoraction.NewApprovalRequest(operation, recipeID, action, artifact.ID{}))(t)
	decision := testcheck.Must(NewHumanDecision(request, operatoraction.AnswerGrant))(t)
	return store, request, decision
}
