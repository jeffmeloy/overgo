package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestPeerWorkspaceVertical(t *testing.T) {
	fixture := newPeerWorkspaceFixture(t, "peer-gui-vertical", "")
	publishPeerControlFixture(t, fixture, "gui-peer")
	manifestResponse := serveTestRequest(fixture.handler, http.MethodGet, "/workspace/manifest", "")
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(manifestResponse.Body.Bytes(), &manifest); err != nil {
		t.Fatal(err)
	}
	peerEnabled := false
	for _, tab := range manifest.Tabs {
		if tab.ID == "peers" {
			peerEnabled = tab.Enabled
		}
	}
	module := serveTestRequest(fixture.handler, http.MethodGet, "/mod/peers.js", "")
	for _, capability := range []string{
		"Inventory, peer labels, and capacity", "Peer detail", "peer labels", "Placement rules", "Staging, attempts, and logs",
		"/peers/enroll", "/peers/state", "/peers/placement", "/peers/reconcile", "/peers/evidence",
	} {
		if !strings.Contains(module.Body.String(), capability) {
			t.Errorf("peer workspace lacks %q", capability)
		}
	}
	if manifestResponse.Code != http.StatusOK || module.Code != http.StatusOK || !peerEnabled {
		t.Fatalf("peer workspace manifest=%d module=%d enabled=%t", manifestResponse.Code, module.Code, peerEnabled)
	}
}

func TestPeerWorkspaceUsesCommonForm(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	module := serveTestRequest(handler, http.MethodGet, "/mod/peers.js", "").Body.String()
	for _, token := range []string{
		"overgo.schemaForm", "/workspace/schema?id=peer-enrollment", "/workspace/schema?id=peer-placement",
		"enrollmentForm.validate()", "placementForm.validate()", "markSaved()", "dispose()",
	} {
		if !strings.Contains(module, token) {
			t.Errorf("peer workspace common form missing %q", token)
		}
	}
	for _, id := range []string{"peer-enrollment", "peer-placement"} {
		response := serveTestRequest(handler, http.MethodGet, "/workspace/schema?id="+id, "")
		if response.Code != http.StatusOK {
			t.Fatalf("peer schema %s status=%d body=%s", id, response.Code, response.Body.String())
		}
	}
}

func TestPeerWorkspaceUsesGlobalOperations(t *testing.T) {
	module := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/mod/peers.js", "").Body.String()
	for _, token := range []string{"openGlobalOperation", `url.searchParams.set("operation"`, "PopStateEvent", "overgo.runtimeEvents.subscribe"} {
		if !strings.Contains(module, token) {
			t.Errorf("peer workspace global operation integration missing %q", token)
		}
	}
	for _, forbidden := range []string{"setInterval", "overgo.poller", `api.get("/operations`} {
		if strings.Contains(module, forbidden) {
			t.Errorf("peer workspace duplicates operation authority with %q", forbidden)
		}
	}
}

func TestPeerWorkspaceSSE(t *testing.T) {
	fixture := newPeerWorkspaceFixture(t, "peer-gui-sse", "")
	ctx, cancel := context.WithCancel(context.Background())
	recorder := &countingRecorder{ResponseRecorder: httptest.NewRecorder(), flushes: make(chan struct{}, peerWorkspaceLimit)}
	done := make(chan struct{})
	go func() {
		fixture.handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/peers/stream", nil).WithContext(ctx))
		close(done)
	}()
	for range 2 {
		<-recorder.flushes
	}
	cancel()
	<-done
	if recorder.Header().Get("Content-Type") != "text/event-stream" ||
		!strings.Contains(recorder.Body.String(), "event: peer.inventory") ||
		!strings.Contains(recorder.Body.String(), "event: operation.snapshot") {
		t.Fatalf("peer SSE headers=%v body=%s", recorder.Header(), recorder.Body.String())
	}
}
