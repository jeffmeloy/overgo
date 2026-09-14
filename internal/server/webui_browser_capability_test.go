package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserAgentCapabilityRefresh(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": capability refresh runs through cmd/webui-lane")
	}
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 45*time.Second, errors.New("agent capability refresh did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL+"/app.html#chat")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(ctx, `!!document.querySelector('#panel-chat .composer')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `!overgo.capabilities().modes.find(mode => mode.id === 'agent').enabled`)
	ensureLaneAgent(t, server.URL)
	for index, state := range []string{"active", "paused", "active"} {
		if index != 0 {
			response := serveTestRequest(fixture.handler, http.MethodPost, "/agents/state", `{"name":"lane-agent","state":"`+state+`"}`)
			if response.Code != http.StatusOK {
				t.Fatalf("agent state %s: %d %s", state, response.Code, response.Body.String())
			}
		}
		assertBrowserPredicate(t, ctx, browser, `(() => { window.previousComposer = document.querySelector('#panel-chat .composer'); overgo.openConversation(null); return true; })()`)
		if err := browser.Eventually(ctx, `!!document.querySelector('#panel-chat .composer') && document.querySelector('#panel-chat .composer') !== previousComposer`); err != nil {
			t.Fatal(err)
		}
		predicate := `overgo.capabilities().modes.find(mode => mode.id === 'agent').enabled && !!document.querySelector('.composer select[aria-label="mode"] option[value="agent"]')`
		if state == "paused" {
			predicate = `!overgo.capabilities().modes.find(mode => mode.id === 'agent').enabled && !document.querySelector('.composer select[aria-label="mode"] option[value="agent"]')`
		}
		assertBrowserPredicate(t, ctx, browser, predicate)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const get=overgo.api.get;
      overgo.api.get=async function(path,options){if(path==='/workspace/manifest')return new Promise((resolve,reject)=>{window.rejectCapabilityManifest=reject;window.releaseCapabilityManifest=async()=>resolve(await get.call(this,path,options));});return get.call(this,path,options);};
      window.pendingCapabilityOpen=overgo.openConversation(null);return true;
    })()`)
	if err := browser.Eventually(ctx, `typeof releaseCapabilityManifest === 'function'`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash='agent'; return true; })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector('#panel-agent.active')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(async () => { await releaseCapabilityManifest(); await pendingCapabilityOpen; return location.hash === '#agent' && !!document.querySelector('#panel-agent.active'); })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => { window.pendingCapabilityOpen=overgo.openConversation(null); location.hash='recipe'; return true; })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector('#panel-recipe.active')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(async () => { rejectCapabilityManifest(new Error('obsolete manifest failure')); await pendingCapabilityOpen; return location.hash === '#recipe' && !document.querySelector('#panel-chat').textContent.includes('obsolete manifest failure'); })()`)
}
