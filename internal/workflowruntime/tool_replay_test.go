package workflowruntime

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAToolWorkflowRefusesChangedArgumentsUnderAKeyThatRan holds a workflow key to the calls it ran (SQA-328): the
// operation was named by the key alone and a completed receipt replayed by operation and node, so item=B under the key
// item=A ran under published B's inputs bound to A's outputs with no tool called. B under A's key is refused naming
// its step, with no transport call; A again still replays.
func TestAToolWorkflowRefusesChangedArgumentsUnderAKeyThatRan(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		response.Header().Set("Content-Type", "application/json")
		response.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()
	inspect := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "state.inspect", Description: "Runtime workflow test tool.", Effect: agenttool.EffectInspection,
		Arguments: []agenttool.Field{{Name: "item", Kind: agenttool.FieldString, Required: true}},
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/inspect"},
	}))(t)
	model := testutil.ArtifactID(t, artifact.KindModel, "tool-replay-model")
	workflow := testcheck.Must(agenttool.CompileToolWorkflow(agenttool.ToolWorkflowProposal{Model: model, Steps: []agenttool.ToolWorkflowStep{
		{ID: "inspect", Manual: inspect.ID},
	}}, []agenttool.Manual{inspect}))(t)
	store := overgodbtest.Open(t, t.TempDir())
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "tool-replay/model", Artifacts: []artifact.Descriptor{{ID: model}}}))(t)
	run := func(item string) error {
		_, err := ExecuteToolWorkflow(t.Context(), store, workflow, agenttool.NewOperatorExecutor(), "the-key",
			map[recipe.NodeID]json.RawMessage{"inspect": json.RawMessage(`{"item":"` + item + `"}`)})
		return err
	}
	testcheck.NoError(t, run("A"))
	ran := calls.Load()
	if err := run("B"); err == nil || !strings.Contains(err.Error(), "inspect") || calls.Load() != ran {
		t.Fatalf("item=B under the key item=A ran ended %v after %d transport calls, want it refused naming the step with none", err, calls.Load()-ran)
	}
	testcheck.NoError(t, run("A"))
}
