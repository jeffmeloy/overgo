package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestOperatorDecisionInteractionContract pins the one-view contract: the
// pending-decision view lists exactly the committed approval requests still
// awaiting a decision with their exact tool and argument surface, deciding
// removes a row only because the durable decision resolves, and the causal
// timeline orders committed receipts and decisions by exact introduction
// sequence while citing canonical identities — no view state, no second
// decision authority.
func TestOperatorDecisionInteractionContract(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	ctx := t.Context()
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, "decision-operation")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "decision-recipe")
	request := testcheck.Must(operatoraction.NewApprovalRequest(operationID, recipeID, operatoraction.Action{
		Code: "retry", Argv: []string{"--attempt", "2"},
	}, artifact.ID{}))(t)
	requestContent := testcheck.Must(request.Content())(t)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "operator/approval/" + request.ID.String(),
		Artifacts: []artifact.Descriptor{{ID: operationID}, {ID: recipeID}},
		Contents:  []artifact.Content{requestContent},
	}); err != nil {
		t.Fatal(err)
	}
	receipt := StageReceipt{
		Recipe: recipeID, Node: "respond", Operation: operationID,
		Attempt: 1, State: StageAdmitted,
	}
	published := testcheck.Must(PublishStageReceipt(ctx, store, receipt, nil, nil))(t)

	pending := testcheck.Must(PendingOperatorDecisions(ctx, store, 10))(t)
	if len(pending) != 1 || pending[0].Operation != operationID || pending[0].Request != request.ID ||
		pending[0].Tool != "retry" || len(pending[0].Arguments) != 2 {
		t.Fatalf("pending decisions = %+v", pending)
	}

	decision := testcheck.Must(NewHumanDecision(request, operatoraction.AnswerGrant))(t)
	testcheck.NoError(t, PublishHumanDecision(ctx, store, request, decision))
	pending = testcheck.Must(PendingOperatorDecisions(ctx, store, 10))(t)
	if len(pending) != 0 {
		t.Fatalf("decided request stayed pending: %+v", pending)
	}

	timeline := testcheck.Must(DeriveOperatorTimeline(ctx, store, operationID, 10))(t)
	if len(timeline) != 2 || timeline[0].Kind != "stage-receipt" || timeline[1].Kind != "human-decision" {
		t.Fatalf("timeline = %+v", timeline)
	}
	if timeline[0].Sequence >= timeline[1].Sequence || timeline[0].Artifact != published.ID ||
		timeline[1].Artifact != decision.ID {
		t.Fatalf("timeline order or citations = %+v", timeline)
	}
	if timeline[1].Cites[0] != request.ID {
		t.Fatalf("decision event does not cite its request: %+v", timeline[1])
	}
	again, err := DeriveOperatorTimeline(ctx, store, operationID, 10)
	if err != nil || len(again) != len(timeline) ||
		again[0].Artifact != timeline[0].Artifact || again[0].Sequence != timeline[0].Sequence {
		t.Fatalf("timeline is not idempotent: %+v vs %+v (%v)", again, timeline, err)
	}
}
