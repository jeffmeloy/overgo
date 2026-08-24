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
	})
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
	for step := 0; step < maxSessionSteps; step++ {
		if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(fmt.Sprintf(`{"step":%d}`, step)), false); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
	}
	if _, err := coordinator.Propose(ctx, session, "probe.read", json.RawMessage(`{}`), false); err == nil ||
		!strings.Contains(err.Error(), "step bound") {
		t.Fatalf("unbounded session: %v", err)
	}
}
