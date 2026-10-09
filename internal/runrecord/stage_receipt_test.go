package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/invocation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

const stageReceiptTestFailure = "execution_failed"

func TestStageReceiptIdentity(t *testing.T) {
	fixture := StageReceipt{
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "stage-receipt-recipe"),
		Node:   "execute", Operation: testutil.ArtifactID(t, artifact.KindEvidence, "stage-receipt-operation"),
		Attempt: 1, State: StageCompleted,
		Inputs: []StageBinding{{Port: recipe.PortName("input"), Artifacts: []artifact.ID{
			testutil.ArtifactID(t, artifact.KindFile, "stage-receipt-input"),
		}}},
	}
	first, err := NewStageReceipt(fixture)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewStageReceipt(fixture)
	if err != nil || first.ID != second.ID {
		t.Fatalf("stage receipt identity = (%s, %s, %v)", first.ID, second.ID, err)
	}
	fixture.State, fixture.Failure = StageFailed, stageReceiptTestFailure
	if failed, err := NewStageReceipt(fixture); err != nil || failed.ID == first.ID {
		t.Fatalf("failed stage identity = (%s, %v)", failed.ID, err)
	}
}

// TestStageAdmittedCrashRecovers closes the admitted-crash finding: a
// stage persisted as admitted whose runner died before persisting
// running must accept a next-attempt admission, exactly as running,
// waiting, and failed stages do.
func TestStageAdmittedCrashRecovers(t *testing.T) {
	base := StageReceipt{
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "stage-admitted-recipe"),
		Node:   "execute", Operation: testutil.ArtifactID(t, artifact.KindEvidence, "stage-admitted-operation"),
		Attempt: 1, State: StageAdmitted,
	}
	previous, err := NewStageReceipt(base)
	if err != nil {
		t.Fatal(err)
	}
	retry := base
	retry.Attempt, retry.State = 2, StageAdmitted
	if !stageTransition(previous, retry) {
		t.Fatal("admitted stage refused a recovery admission")
	}
	same := base
	same.State = StageCompleted
	if stageTransition(previous, same) {
		t.Fatal("admitted stage completed without running")
	}
	skipped := base
	skipped.Attempt, skipped.State = 3, StageAdmitted
	if stageTransition(previous, skipped) {
		t.Fatal("recovery skipped an attempt number")
	}
}

// earlierStageReceipt commits one receipt the way the contract before
// invocation bindings wrote it, bound at its operation node's alias, and
// returns the identity the store holds it by.
func earlierStageReceipt(t *testing.T, store artifact.Repository, value StageReceipt) artifact.ID {
	t.Helper()
	receipt, err := NewStageReceipt(value)
	if err != nil {
		t.Fatal(err)
	}
	present, err := stageReceiptCodec.Content(receipt)
	if err != nil {
		t.Fatal(err)
	}
	earlier, err := artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: StageReceiptMediaType, Schema: stageReceiptEarlierSchema}.ContentBytes(present.Data)
	if err != nil {
		t.Fatal(err)
	}
	if earlier.Descriptor.ID != receipt.ID {
		t.Fatalf("an earlier receipt of the same bytes is %s, the present one %s; a receipt's identity is its bytes'", earlier.Descriptor.ID, receipt.ID)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/earlier-stage-receipt/" + receipt.ID.String(), Contents: []artifact.Content{earlier},
		Aliases: []artifact.AliasBinding{artifact.AliasMove(stageReceiptAlias(receipt.Operation, receipt.Node), receipt.ID, artifact.ID{})},
	}); err != nil {
		t.Fatal(err)
	}
	return receipt.ID
}

// TestStageReceiptOfTheEarlierFormIsRead holds a store's receipts of the
// contract before invocation bindings readable. A store keeps the receipts of
// every request it served under that contract at the aliases a repeat of the
// request resolves, and a reader that took only the present contract refused
// the repeat before any stage ran. An earlier receipt is read as it stands,
// under its own identity; the next receipt of its stage names it as previous
// and takes the alias from it; the operation's timeline walks back through
// it; and a document of the earlier contract that binds an invocation, which
// that contract could not write, is refused.
func TestStageReceiptOfTheEarlierFormIsRead(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "earlier-receipt-recipe")
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "earlier-receipt-operation")
	output := testutil.ArtifactID(t, artifact.KindOutput, "earlier-receipt-output")
	for _, id := range []artifact.ID{recipeID, operation, output} {
		testutil.PublishArtifact(t, store, id)
	}
	written := StageReceipt{
		Recipe: recipeID, Node: "decode", Operation: operation, Attempt: 1, State: StageCompleted,
		Inputs:  []StageBinding{{Port: "features"}},
		Outputs: []StageBinding{{Port: "image", Artifacts: []artifact.ID{output}}},
	}
	earlier := earlierStageReceipt(t, store, written)
	read, found, err := ResolveStageReceipt(ctx, store, operation, "decode")
	if err != nil || !found || read.ID != earlier || read.Attempt != 1 || read.State != StageCompleted ||
		len(read.Outputs) != 1 || len(read.Outputs[0].Artifacts) != 1 || read.Outputs[0].Artifacts[0] != output || read.Invocation != nil {
		t.Fatalf("the earlier receipt read as %+v (found %t, %v); want it as written, under %s", read, found, err, earlier)
	}
	next, err := PublishStageReceipt(ctx, store, StageReceipt{Recipe: recipeID, Node: "decode", Operation: operation, Attempt: 2, State: StageAdmitted}, nil, nil)
	if err != nil || next.Previous != earlier {
		t.Fatalf("the stage's next receipt names %s as previous (%v); want the earlier receipt %s", next.Previous, err, earlier)
	}
	tip, found, err := ResolveStageReceipt(ctx, store, operation, "decode")
	if err != nil || !found || tip.ID != next.ID {
		t.Fatalf("the stage's alias resolves %s (found %t, %v); want the next receipt %s", tip.ID, found, err, next.ID)
	}
	timeline, err := DeriveOperatorTimeline(ctx, store, operation, 8)
	if err != nil {
		t.Fatal(err)
	}
	walked := map[artifact.ID]bool{}
	for _, event := range timeline {
		walked[event.Artifact] = true
	}
	if !walked[earlier] || !walked[next.ID] {
		t.Fatalf("the operation's timeline is %+v; want both the earlier receipt and the one after it", timeline)
	}

	bound := written
	bound.Node = "promote"
	bound.Invocation = &invocation.ReceiptBinding{
		Boundary: invocation.BoundaryInternal, Kind: invocation.MutationPromotion, Action: "model.promote",
		Subject:       testutil.ArtifactID(t, artifact.KindRecipe, "earlier-receipt-subject"),
		Arguments:     testutil.ArtifactID(t, artifact.KindEvidence, "earlier-receipt-arguments"),
		Effect:        testutil.ArtifactID(t, artifact.KindEvidence, "earlier-receipt-effect"),
		Preflight:     testutil.ArtifactID(t, artifact.KindEvidence, "earlier-receipt-preflight"),
		Inspection:    testutil.ArtifactID(t, artifact.KindEvidence, "earlier-receipt-inspection"),
		Ceiling:       testutil.ArtifactID(t, artifact.KindRecipe, "earlier-receipt-ceiling"),
		Authority:     testutil.ArtifactID(t, artifact.KindEvidence, "earlier-receipt-decision"),
		CausalContext: testutil.ArtifactID(t, artifact.KindEvidence, "earlier-receipt-causal"),
		Head:          artifact.CommitID{1},
	}
	earlierStageReceipt(t, store, bound)
	if read, found, err := ResolveStageReceipt(ctx, store, operation, "promote"); err == nil {
		t.Fatalf("an earlier-contract document that binds an invocation was read as %+v (found %t)", read, found)
	}
}
