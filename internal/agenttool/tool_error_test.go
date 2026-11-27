package agenttool

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/strictjson"
)

// TestAToolResultMarkedAnErrorFailsTheCall holds an MCP tool call to the result its server marks: isError true, the
// shape cmd/loop's own tool server writes when a worker tool fails, fails the call with the tool's text, so the
// coordinator records a mutation failed and credits no inspection; false, or the flag omitted, is the tool's result.
// The result had been taken whole whatever it said (SQA-508).
func TestAToolResultMarkedAnErrorFailsTheCall(t *testing.T) {
	t.Parallel()
	for _, each := range []struct {
		name, result string
		fails        bool
	}{
		{name: "marked an error", result: `{"content":[{"type":"text","text":"edit refused: the file is outside the step"}],"isError":true}`, fails: true},
		{name: "marked no error", result: `{"content":[{"type":"text","text":"done"}],"isError":false}`},
		{name: "unmarked", result: `{"content":[{"type":"text","text":"done"}]}`},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
			var rpc mcpRequest
			if err := strictjson.Decode(request.Body, &rpc); err != nil {
				response.WriteHeader(http.StatusBadRequest)
				return
			}
			switch rpc.Method {
			case mcpInitializeMethod:
				fmt.Fprintf(response, `{"jsonrpc":"2.0","id":%q,"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fixture","version":"1"}}}`, rpc.ID)
			case mcpInitializedMethod:
				response.WriteHeader(http.StatusAccepted)
			default:
				fmt.Fprintf(response, `{"jsonrpc":"2.0","id":%q,"result":%s}`, rpc.ID, each.result)
			}
		}))
		manual := inspectionManual(t, "edit.remote", Transport{Kind: TransportMCPHTTP, URL: server.URL, Target: "edit", Protocol: "2025-06-18"})
		result, err := NewOperatorExecutor().Invoke(t.Context(), manual, json.RawMessage(`{"pattern":"main.go"}`))
		server.Close()
		switch {
		case each.fails && (err == nil || !strings.Contains(err.Error(), "the file is outside the step")):
			t.Errorf("a result %s answered %s, %v; want the call failed with the tool's text", each.name, result, err)
		case !each.fails && (err != nil || string(result) != each.result):
			t.Errorf("a result %s answered %s, %v; want the result", each.name, result, err)
		}
	}
}
