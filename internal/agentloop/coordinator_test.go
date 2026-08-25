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
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const fixtureSessionSteps = 2

func coordinatorFixture(t *testing.T) (*Coordinator, *overgodb.Store) {
	t.Helper()
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "agent-loop-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "agent-loop-model")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "agent-loop/identity", Artifacts: []artifact.Descriptor{{ID: recipeID}},
	}); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/read", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"seen":true}`)) })
	mux.HandleFunc("/write", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"changed":true}`)) })
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	inspect, err := agenttool.NewManual(agenttool.Manual{
		Name: "probe.read", Description: "Read state for the loop fixture.",
		Effect:    agenttool.EffectInspection,
		Arguments: []agenttool.Field{{Name: "step", Kind: agenttool.FieldInteger}},
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/read"},
	})
	if err != nil {
		t.Fatal(err)
	}
	mutate, err := agenttool.NewManual(agenttool.Manual{
		Name: "probe.write", Description: "Change state for the loop fixture.",
		Effect:    agenttool.EffectMutation,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/write"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{inspect, mutate}); err != nil {
		t.Fatal(err)
	}
	executor := agenttool.NewOperatorExecutor()
	coordinator, err := New(store, executor, Identity{
		Recipe: recipeID, Model: modelID, Node: recipe.NodeID("respond"),
	}, fixtureSessionSteps)
	if err != nil {
		t.Fatal(err)
	}
	return coordinator, store
}

// TestCoordinatorGatesMutationBehindInspectionAndApproval pins the
// admission ladder: unregistered names refuse, mutation refuses before
// inspection, refuses without approval, and runs after both -- with
// every admitted step durably chained.
func TestCoordinatorGatesMutationBehindInspectionAndApproval(t *testing.T) {
	ctx := context.Background()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "agent-session-1"}
	if _, err := coordinator.Propose(ctx, session, "probe.ghost", json.RawMessage(`{}`), false); err == nil ||
		!strings.Contains(err.Error(), "unregistered") {
		t.Fatalf("unregistered tool admitted: %v", err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`), true); err == nil ||
		!strings.Contains(err.Error(), "inspection") {
		t.Fatalf("mutation admitted before inspection: %v", err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`), false); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`), false); err == nil ||
		!strings.Contains(err.Error(), "approval") {
		t.Fatalf("mutation admitted without approval: %v", err)
	}
	// The approve FLAG alone is not authority: without a committed
	// decision the mutation refuses; a decision bound to DIFFERENT
	// argument bytes refuses too; only the exact-bound grant admits.
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`), true); err == nil ||
		!strings.Contains(err.Error(), "no committed approval decision") {
		t.Fatalf("mutation admitted on the flag without a committed decision: %v", err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{"other":1}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`), true); err == nil ||
		!strings.Contains(err.Error(), "different tool or arguments") {
		t.Fatalf("mutation admitted under a decision for different arguments: %v", err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	result, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`), true)
	if err != nil || string(result) != `{"changed":true}` {
		t.Fatalf("approved mutation = %s, %v", result, err)
	}
	if session.Steps != 2 || !session.Inspected {
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
	ctx := context.Background()
	coordinator, _ := coordinatorFixture(t)
	session := &Session{ID: "agent-session-bound"}
	for step := 0; step < coordinator.maxSteps; step++ {
		if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(fmt.Sprintf(`{"step":%d}`, step)), false); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
	}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`), false); err == nil ||
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
	ctx := context.Background()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "identity-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`), false); err != nil {
		t.Fatal(err)
	}
	if !session.Inspected {
		t.Fatal("first inspection did not set the inspection fact")
	}
	// Rebind the "probe.read" name to a mutation manual -- the same name,
	// a different effect and identity.
	mutated, err := agenttool.NewManual(agenttool.Manual{
		Name: "probe.read", Description: "Now a mutation under the old name.",
		Effect: agenttool.EffectMutation, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{mutated}); err != nil {
		t.Fatal(err)
	}
	restored, err := coordinator.RestoreSession(ctx, "identity-session")
	if err != nil {
		t.Fatal(err)
	}
	if restored.Steps != 1 || !restored.Inspected {
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
	ctx := context.Background()
	coordinator, store := coordinatorFixture(t)
	session := &Session{ID: "receipt-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`), false); err != nil {
		t.Fatal(err)
	}
	inspectOperation, err := MutationReceiptOperation("receipt-session-step-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, found, err := runrecord.ResolveStageReceipt(ctx, store, inspectOperation, coordinator.identity.Node); err != nil || found {
		t.Fatalf("inspection carried a receipt: found=%t err=%v", found, err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.write", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.write", json.RawMessage(`{}`), true); err != nil {
		t.Fatal(err)
	}
	operation, err := MutationReceiptOperation("receipt-session-step-2")
	if err != nil {
		t.Fatal(err)
	}
	tip, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, coordinator.identity.Node)
	if err != nil || !found {
		t.Fatalf("mutation receipt chain absent: found=%t err=%v", found, err)
	}
	if tip.State != runrecord.StageCompleted || len(tip.Outputs) != 1 || tip.Outputs[0].Port != "result" {
		t.Fatalf("receipt tip = %+v, want completed with a bound result", tip)
	}
	states := []runrecord.StageState{tip.State}
	for previous := tip.Previous; previous.Valid(); {
		parsed, err := readStageReceipt(t, ctx, store, previous)
		if err != nil {
			t.Fatal(err)
		}
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
	ctx := context.Background()
	coordinator, store := coordinatorFixture(t)
	broken, err := agenttool.NewManual(agenttool.Manual{
		Name: "probe.break", Description: "Fail for the receipt fixture.",
		Effect:    agenttool.EffectMutation,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: "http://127.0.0.1:9/never"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{broken}); err != nil {
		t.Fatal(err)
	}
	session := &Session{ID: "failing-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`), false); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.ApproveMutation(ctx, session, "probe.break", json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := coordinator.Propose(ctx, session, "probe.break", json.RawMessage(`{}`), true); err == nil {
		t.Fatal("broken mutation transport succeeded")
	}
	operation, err := MutationReceiptOperation("failing-session-step-2")
	if err != nil {
		t.Fatal(err)
	}
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
