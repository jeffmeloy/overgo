package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const fixtureSessionSteps = 2

func coordinatorFixture(t *testing.T) (*Coordinator, *overgodb.Store) {
	t.Helper()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "agent-loop-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "agent-loop-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "agent-loop/identity", Artifacts: []artifact.Descriptor{{ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /read", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"seen":true}`)) })
	mux.HandleFunc("POST /write", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"changed":true}`)) })
	mux.HandleFunc("POST /break", func(w http.ResponseWriter, _ *http.Request) { http.Error(w, "broken", http.StatusInternalServerError) })
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	inspect := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "probe.read", Description: "Read state for the loop fixture.",
		Effect:    agenttool.EffectInspection,
		Arguments: []agenttool.Field{{Name: "step", Kind: agenttool.FieldInteger}},
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/read"},
	}))(t)
	mutate := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "probe.write", Description: "Change state for the loop fixture.",
		Effect:    agenttool.EffectMutation,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/write"},
	}))(t)
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{inspect, mutate}); err != nil {
		t.Fatal(err)
	}
	executor := agenttool.NewOperatorExecutor()
	coordinator := testcheck.Must(New(store, executor, Identity{
		Recipe: recipeID, Model: modelID, Node: recipe.NodeID("respond"),
	}, fixtureSessionSteps))(t)
	return coordinator, store
}

// previewedOperation mirrors the operator flow: read the decision preview
// first, then grant by naming exactly the operation it projected.
func previewedOperation(t *testing.T, coordinator *Coordinator, session *Session, name string) artifact.ID {
	t.Helper()
	preview := testcheck.Must(coordinator.PreviewMutationDecision(t.Context(), session, name, json.RawMessage(`{}`)))(t)
	return preview.Operation
}

// TestCoordinatorGatesMutationBehindInspectionAndApproval pins the
// admission ladder: unregistered names refuse, mutation refuses before
// inspection, refuses without approval, and runs after both -- with
// every admitted step durably chained.
func TestCoordinatorGatesMutationBehindInspectionAndApproval(t *testing.T) {
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "agent-session-1"}
	if _, err := coordinator.Propose(ctx, session, "probe.ghost", json.RawMessage(`{}`)); err == nil ||
		!strings.Contains(err.Error(), "unregistered") {
		t.Fatalf("unregistered tool admitted: %v", err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`)); err == nil ||
		!strings.Contains(err.Error(), "preflight") {
		t.Fatalf("mutation admitted before inspection: %v", err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`),
		previewedOperation(t, coordinator, session, "probe.write")); err == nil ||
		!strings.Contains(err.Error(), "inspection") {
		t.Fatalf("mutation preflight admitted before inspection: %v", err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`)); err == nil ||
		!strings.Contains(err.Error(), "preflight") {
		t.Fatalf("mutation admitted without approval: %v", err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`), session.Interaction); err == nil ||
		!strings.Contains(err.Error(), "does not name the previewed operation") {
		t.Fatalf("grant naming the wrong operation admitted: %v", err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`),
		previewedOperation(t, coordinator, session, "probe.write")); err != nil {
		t.Fatal(err)
	}
	result, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`))
	if err != nil || string(result) != `{"changed":true}` {
		t.Fatalf("approved mutation = %s, %v", result, err)
	}
	if session.Steps != 2 || session.Inspection.Valid() || session.InspectionEffect.Valid() {
		t.Fatalf("session = %+v", session)
	}
	interaction, found, err := runrecord.ResolveInteraction(ctx, store, "agent-session-1-step-2")
	if err != nil || !found {
		t.Fatalf("durable step = (%t, %v)", found, err)
	}
	if interaction.ID != session.Interaction || !interaction.Parent.Valid() {
		t.Fatalf("interaction chain = %+v, session tip %s", interaction, session.Interaction)
	}
}

// TestCoordinatorBoundsSessionSteps pins the step bound: the session
// halts at a typed refusal instead of stepping forever.
func TestCoordinatorBoundsSessionSteps(t *testing.T) {
	ctx := t.Context()
	coordinator, _ := coordinatorFixture(t)
	session := &Session{ID: "agent-session-bound"}
	for step := 0; step < coordinator.maxSteps; step++ {
		if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(fmt.Sprintf(`{"step":%d}`, step))); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
	}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`)); err == nil ||
		!strings.Contains(err.Error(), "step bound") {
		t.Fatalf("unbounded session: %v", err)
	}
}

// TestRestoreResolvesToolByExactIdentity pins the reopened finding: a
// restored session's inspection state is decided by the exact manual
// each step ran, so republishing the tool's name alias with a
// different effect cannot rewrite history. It inspects, then the
// registered "probe.read" is superseded by a MUTATION manual under the
// same name; a restore must still see step one as an inspection.
func TestRestoreResolvesToolByExactIdentity(t *testing.T) {
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "identity-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	if !session.Inspection.Valid() || !session.InspectionEffect.Valid() {
		t.Fatal("first inspection did not set the inspection fact")
	}
	// Rebind the "probe.read" name to a mutation manual -- the same name,
	// a different effect and identity.
	mutated := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "probe.read", Description: "Now a mutation under the old name.",
		Effect: agenttool.EffectMutation, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
	}))(t)
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{mutated}); err != nil {
		t.Fatal(err)
	}
	restored := testcheck.Must(coordinator.RestoreSession(ctx, "identity-session"))(t)
	if restored.Steps != 1 || !restored.Inspection.Valid() || !restored.InspectionEffect.Valid() {
		t.Fatalf("restored = %+v, want the step-one inspection preserved despite the alias effect change", restored)
	}
}

// TestMutationReceiptPrecedesExecution pins the reopened finding: a
// mutation persists a durable receipt chain BEFORE it executes and
// closes it after. A succeeding mutation leaves admitted -> running ->
// completed with the manual, arguments, and result bound; a FAILING
// mutation still leaves its chain, terminally failed with the error
// preserved -- the side effect attempt never lacks evidence. An
// inspection carries no receipt.
func TestMutationReceiptPrecedesExecution(t *testing.T) {
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "receipt-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	inspectOperation := testcheck.Must(MutationReceiptOperation("receipt-session-step-1"))(t)
	if _, found, err := runrecord.ResolveStageReceipt(ctx, store, inspectOperation, coordinator.identity.Node); err != nil || found {
		t.Fatalf("inspection carried a receipt: found=%t err=%v", found, err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`),
		previewedOperation(t, coordinator, session, "probe.write")); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	operation := testcheck.Must(MutationReceiptOperation("receipt-session-step-2"))(t)
	tip, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, coordinator.identity.Node)
	if err != nil || !found {
		t.Fatalf("mutation receipt chain absent: found=%t err=%v", found, err)
	}
	if tip.State != runrecord.StageCompleted || len(tip.Outputs) != 1 || tip.Outputs[0].Port != "result" ||
		tip.Invocation == nil || !tip.Invocation.Inspection.Valid() || !tip.Invocation.Authority.Valid() {
		t.Fatalf("receipt tip = %+v, want completed with a bound result", tip)
	}
	states := []runrecord.StageState{tip.State}
	for previous := tip.Previous; previous.Valid(); {
		parsed := testcheck.Must(readStageReceipt(t, ctx, store, previous))(t)
		states = append(states, parsed.State)
		previous = parsed.Previous
	}
	if len(states) != 3 || states[2] != runrecord.StageAdmitted || states[1] != runrecord.StageRunning {
		t.Fatalf("receipt states = %v, want completed <- running <- admitted", states)
	}
}

// TestFailingMutationStillLeavesReceipt pins the failure half: the
// transport errors, the proposal errors, and the receipt chain still
// exists with a terminal failed state carrying the error.
func TestFailingMutationStillLeavesReceipt(t *testing.T) {
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	inspection := testcheck.Must(agenttool.ResolveRegisteredManual(ctx, store, "probe.read"))(t)
	broken := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "probe.break", Description: "Fail for the receipt fixture.",
		Effect:    agenttool.EffectMutation,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: strings.TrimSuffix(inspection.Transport.URL, "/read") + "/break"},
	}))(t)
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{broken}); err != nil {
		t.Fatal(err)
	}
	session := &Session{ID: "failing-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.break", json.RawMessage(`{}`),
		previewedOperation(t, coordinator, session, "probe.break")); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.break", json.RawMessage(`{}`)); err == nil {
		t.Fatal("broken mutation transport succeeded")
	}
	operation := testcheck.Must(MutationReceiptOperation("failing-session-step-2"))(t)
	tip, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, coordinator.identity.Node)
	if err != nil || !found {
		t.Fatalf("failed mutation left no receipt: found=%t err=%v", found, err)
	}
	if tip.State != runrecord.StageFailed || tip.Failure == "" {
		t.Fatalf("receipt tip = %+v, want terminal failed with the error preserved", tip)
	}
}

func readStageReceipt(t *testing.T, ctx context.Context, store *overgodb.Store, id artifact.ID) (runrecord.StageReceipt, error) {
	t.Helper()
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil || !found {
		t.Fatalf("receipt %s content: found=%t err=%v", id, found, err)
	}
	return runrecord.ParseStageReceipt(content.Data)
}
