package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/invocation"
	"overgo/internal/testutil"
)

func TestMutationRequiresBoundPreflightInspection(t *testing.T) {
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "preflight-operation")
	arguments := testutil.ArtifactID(t, artifact.KindEvidence, "preflight-arguments")
	selection, err := dataset.SelectInteractions(dataset.InteractionSelectionBounds{
		MaxTokens: 1, MaxBytes: 1, MaxDocuments: 1, MaxDepth: 1, MaxResults: 1,
	}, artifact.CommitID{1}, []dataset.InteractionSelectionSource{{
		Source: arguments, CausalRoot: operation, Tokens: 1, Bytes: 1, Documents: 1, Depth: 1,
	}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	boundary := AttemptStimulusBoundary{
		Version: artifact.InitialDocumentVersion, Operation: operation, Attempt: 1,
		Manual: testutil.ArtifactID(t, artifact.KindRecipe, "preflight-manual"),
		Class:  invocation.ClassMutation, Arguments: arguments,
		Effect:    testutil.ArtifactID(t, artifact.KindEvidence, "preflight-effect"),
		Ceiling:   testutil.ArtifactID(t, artifact.KindRecipe, "preflight-ceiling"),
		Selection: selection,
	}
	if err := canonicalizeAttemptStimulus(&boundary); err == nil {
		t.Fatal("mutation without an action-bound inspection was admitted")
	}
	boundary.Inspection = testutil.ArtifactID(t, artifact.KindEvidence, "preflight-inspection")
	boundary.InspectionEffect = testutil.ArtifactID(t, artifact.KindEvidence, "preflight-inspection-effect")
	boundary.CausalContext = boundary.Inspection
	boundary.Prior = boundary.CausalContext
	if err := canonicalizeAttemptStimulus(&boundary); err != nil {
		t.Fatalf("complete action-bound preflight refused: %v", err)
	}
	changed := boundary
	changed.CausalContext = testutil.ArtifactID(t, artifact.KindEvidence, "different-context")
	if err := canonicalizeAttemptStimulus(&changed); err == nil {
		t.Fatal("preflight detached from its inspected causal context")
	}
}

func TestRSIMutationEffectApprovalReceiptRatchet(t *testing.T) {
	binding := invocation.ReceiptBinding{
		Boundary: invocation.BoundaryInternal, Kind: invocation.MutationPromotion, Action: "model.promote",
		Subject:       testutil.ArtifactID(t, artifact.KindRecipe, "receipt-subject"),
		Arguments:     testutil.ArtifactID(t, artifact.KindEvidence, "receipt-arguments"),
		Effect:        testutil.ArtifactID(t, artifact.KindEvidence, "receipt-effect"),
		Preflight:     testutil.ArtifactID(t, artifact.KindEvidence, "receipt-preflight"),
		Inspection:    testutil.ArtifactID(t, artifact.KindEvidence, "receipt-inspection"),
		Ceiling:       testutil.ArtifactID(t, artifact.KindRecipe, "receipt-ceiling"),
		Authority:     testutil.ArtifactID(t, artifact.KindEvidence, "receipt-decision"),
		CausalContext: testutil.ArtifactID(t, artifact.KindEvidence, "receipt-causal"),
		Head:          artifact.CommitID{1},
	}
	receipt := StageReceipt{
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "receipt-recipe"),
		Node:   "promote", Operation: testutil.ArtifactID(t, artifact.KindEvidence, "receipt-operation"),
		Attempt: 1, State: StageAdmitted, Invocation: &binding,
	}
	identified, err := NewStageReceipt(receipt)
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*invocation.ReceiptBinding){
		"effect":     func(value *invocation.ReceiptBinding) { value.Effect = artifact.ID{} },
		"preflight":  func(value *invocation.ReceiptBinding) { value.Preflight = artifact.ID{} },
		"inspection": func(value *invocation.ReceiptBinding) { value.Inspection = artifact.ID{} },
		"ceiling":    func(value *invocation.ReceiptBinding) { value.Ceiling = artifact.ID{} },
		"approval":   func(value *invocation.ReceiptBinding) { value.Authority = artifact.ID{} },
		"head":       func(value *invocation.ReceiptBinding) { value.Head = artifact.CommitID{} },
	} {
		t.Run(name, func(t *testing.T) {
			forged := receipt
			copy := binding
			mutate(&copy)
			forged.Invocation = &copy
			if _, err := NewStageReceipt(forged); err == nil {
				t.Fatalf("receipt without %s authority was admitted", name)
			}
		})
	}
	next := identified
	next.State = StageRunning
	forged := binding
	forged.Arguments = testutil.ArtifactID(t, artifact.KindEvidence, "substituted-arguments")
	next.Invocation = &forged
	if stageTransition(identified, next) {
		t.Fatal("receipt transition substituted invocation authority")
	}
}
