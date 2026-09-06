package agentloop

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// countingCoordinator: a coordinator whose one inspection tool counts the
// HTTP executions it receives, so a replay that reaches the tool is visible.
func countingCoordinator(t *testing.T) (*Coordinator, *overgodb.Store, *atomic.Int32) {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "child-attempt-recipe")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "child-attempt/identity", Artifacts: []artifact.Descriptor{{ID: recipeID}}}); err != nil {
		t.Fatal(err)
	}
	var executions atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		executions.Add(1)
		w.Write([]byte(`{"seen":true}`))
	}))
	t.Cleanup(server.Close)
	inspect, err := agenttool.NewManual(agenttool.Manual{
		Name: "count.read", Description: "Count executions for the replay fixture.",
		Effect:    agenttool.EffectInspection,
		Arguments: []agenttool.Field{{Name: "step", Kind: agenttool.FieldInteger}},
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{inspect}); err != nil {
		t.Fatal(err)
	}
	coordinator, err := New(store, agenttool.NewOperatorExecutor(), Identity{
		Recipe: recipeID, Model: testutil.ArtifactID(t, artifact.KindModel, "child-attempt-model"), Node: recipe.NodeID("respond"),
	}, 4)
	if err != nil {
		t.Fatal(err)
	}
	return coordinator, store, &executions
}

// TestToolCallsReplayAsChildAttempts pins: every executed tool call is a
// child attempt in the session's log with its own interaction receipt; a
// resumed session reads the completed receipts and executes only the calls
// the log has not recorded; the tool is reached exactly once per distinct
// call across both invocations.
func TestToolCallsReplayAsChildAttempts(t *testing.T) {
	ctx := t.Context()
	coordinator, store, executions := countingCoordinator(t)
	calls := []json.RawMessage{json.RawMessage(`{"step":1}`), json.RawMessage(`{"step":2}`)}
	first, err := coordinator.OpenDurableSession(ctx, "child-attempts")
	if err != nil {
		t.Fatal(err)
	}
	var receipts []artifact.ID
	for _, arguments := range calls {
		if _, err := coordinator.Propose(ctx, first, "count.read", arguments); err != nil {
			t.Fatal(err)
		}
		receipts = append(receipts, first.Interaction)
	}
	if executions.Load() != 2 {
		t.Fatalf("first invocation reached the tool %d times, want 2", executions.Load())
	}
	for index, receipt := range receipts {
		ordinal := strconv.Itoa(index + 1)
		child, found, err := first.attempt.Lookup(ctx, store, runrecord.DurableChild, ordinal)
		if err != nil || !found || child.Result != receipt || child.Invocation != 1 {
			t.Fatalf("child %s = %+v found=%t, %v", ordinal, child, found, err)
		}
		interaction, err := runrecord.RequireInteraction(ctx, store, receipt)
		if err != nil || interaction.Response != "child-attempts-step-"+ordinal {
			t.Fatalf("child %s receipt = %+v, %v", ordinal, interaction, err)
		}
	}

	resumed, err := coordinator.OpenDurableSession(ctx, "child-attempts")
	if err != nil {
		t.Fatal(err)
	}
	for index, arguments := range calls {
		result, err := coordinator.Propose(ctx, resumed, "count.read", arguments)
		if err != nil || string(result) != `{"seen":true}` || resumed.Interaction != receipts[index] {
			t.Fatalf("replayed call %d = %s interaction=%s, %v", index+1, result, resumed.Interaction, err)
		}
	}
	if executions.Load() != 2 {
		t.Fatalf("resumed invocation re-executed recorded calls: %d executions", executions.Load())
	}
	if _, err := coordinator.Propose(ctx, resumed, "count.read", json.RawMessage(`{"step":3}`)); err != nil || resumed.Steps != 3 {
		t.Fatalf("new call after replay = steps=%d, %v", resumed.Steps, err)
	}
	if executions.Load() != 3 {
		t.Fatalf("the unrecorded call did not execute exactly once: %d executions", executions.Load())
	}
	if child, found, err := resumed.attempt.Lookup(ctx, store, runrecord.DurableChild, "3"); err != nil || !found || child.Invocation != 2 || child.Result != resumed.Interaction {
		t.Fatalf("third child = %+v found=%t, %v", child, found, err)
	}
}
