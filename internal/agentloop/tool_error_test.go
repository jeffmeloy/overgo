package agentloop

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/testcheck"
)

// TestAToolResultMarkedAnErrorFailsTheCall holds the coordinator to a tool's own failure: an MCP inspection whose server
// answers isError, as cmd/loop's tool server answers a worker tool that failed, fails the call, takes no step and earns
// no inspection credit, and a restart rebuilds none. Read as a result, the failed inspection had been credited, and a
// restart had rebuilt the credit from the recorded step (SQA-508).
func TestAToolResultMarkedAnErrorFailsTheCall(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	replies := map[string]string{
		"initialize": `{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}`,
		"tools/call": `{"content":[{"type":"text","text":"read refused: no such file"}],"isError":true}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var rpc struct {
			ID     string `json:"id"`
			Method string `json:"method"`
		}
		if json.NewDecoder(request.Body).Decode(&rpc) != nil {
			response.WriteHeader(http.StatusBadRequest)
			return
		}
		if reply, held := replies[rpc.Method]; held {
			fmt.Fprintf(response, `{"jsonrpc":"2.0","id":%q,"result":%s}`, rpc.ID, reply)
			return
		}
		response.WriteHeader(http.StatusAccepted)
	}))
	t.Cleanup(server.Close)
	remote := testcheck.Must(agenttool.NewManual(agenttool.Manual{
		Name: "probe.remote", Description: "Read state through a remote tool for the loop fixture.",
		Effect:    agenttool.EffectInspection,
		Transport: agenttool.Transport{Kind: agenttool.TransportMCPHTTP, URL: server.URL, Target: "read", Protocol: "2025-06-18"},
	}))(t)
	testcheck.Must(agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{remote}))(t)
	session := &Session{ID: "remote-session"}
	if _, err := coordinator.Propose(ctx, session, "probe.remote", json.RawMessage(`{}`)); err == nil {
		t.Fatal("an inspection its tool marked an error succeeded")
	}
	if session.Steps != 0 || session.Inspection.Valid() {
		t.Fatalf("a failed inspection left %d steps and inspection %v; want none", session.Steps, session.Inspection)
	}
	restored := testcheck.Must(coordinator.RestoreSession(ctx, "remote-session"))(t)
	if restored.Steps != 0 || restored.Inspection.Valid() {
		t.Fatalf("a restart rebuilt %d steps and inspection %v from a failed inspection; want none", restored.Steps, restored.Inspection)
	}
}
