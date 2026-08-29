package workflowruntime

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestToolWorkflowExecutesCompiledOrderWithDurableRun(t *testing.T) {
	var mu sync.Mutex
	var order []string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		mu.Lock()
		order = append(order, request.URL.Path)
		mu.Unlock()
		response.Header().Set("Content-Type", "application/json")
		response.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()
	inspect := runtimeWorkflowManual(t, "state.inspect", agenttool.EffectInspection, server.URL+"/inspect")
	mutate := runtimeWorkflowManual(t, "state.update", agenttool.EffectMutation, server.URL+"/update")
	model := testutil.ArtifactID(t, artifact.KindModel, "tool-workflow-model")
	workflow, err := agenttool.CompileToolWorkflow(agenttool.ToolWorkflowProposal{Model: model, Steps: []agenttool.ToolWorkflowStep{
		{ID: "inspect", Manual: inspect.ID},
		{ID: "update", Manual: mutate.ID, After: []recipe.NodeID{"inspect"}},
	}}, []agenttool.Manual{inspect, mutate})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "tool-workflow/model", Artifacts: []artifact.Descriptor{{ID: model}},
	}); err != nil {
		t.Fatal(err)
	}
	result, err := ExecuteToolWorkflow(
		t.Context(), store, workflow, agenttool.NewOperatorExecutor(), "test",
		map[recipe.NodeID]json.RawMessage{"inspect": json.RawMessage(`{}`), "update": json.RawMessage(`{}`)},
	)
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(order, []string{"/inspect", "/update"}) || !result.Run.ID.Valid() || len(result.Outputs) != 2 {
		t.Fatalf("order=%v run=%s outputs=%d", order, result.Run.ID, len(result.Outputs))
	}
}

func runtimeWorkflowManual(t *testing.T, name string, effect agenttool.Effect, endpoint string) agenttool.Manual {
	t.Helper()
	manual, err := agenttool.NewManual(agenttool.Manual{
		Name: name, Description: "Runtime workflow test tool.", Effect: effect,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: endpoint},
	})
	if err != nil {
		t.Fatal(err)
	}
	return manual
}
