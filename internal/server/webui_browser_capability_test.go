package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/testcheck"
)

func TestWebUIBrowserAgentCapabilityRefresh(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "capability refresh runs through cmd/webui-lane")
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	ctx, browser, _, _ := openBrowserLeg(t, server.URL+"/app.html#chat")
	defer browser.Close()
	testcheck.NoError(t, browser.Eventually(ctx, `!!document.querySelector('#panel-chat .composer')`))
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
		testcheck.NoError(t, browser.Eventually(ctx, `!!document.querySelector('#panel-chat .composer') && document.querySelector('#panel-chat .composer') !== previousComposer`))
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
	testcheck.NoError(t, browser.Eventually(ctx, `typeof releaseCapabilityManifest === 'function'`))
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash='agent'; return true; })()`)
	testcheck.NoError(t, browser.Eventually(ctx, `!!document.querySelector('#panel-agent.active')`))
	assertBrowserPredicate(t, ctx, browser, `(async () => { await releaseCapabilityManifest(); await pendingCapabilityOpen; return location.hash === '#agent' && !!document.querySelector('#panel-agent.active'); })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => { window.pendingCapabilityOpen=overgo.openConversation(null); location.hash='recipe'; return true; })()`)
	testcheck.NoError(t, browser.Eventually(ctx, `!!document.querySelector('#panel-recipe.active')`))
	assertBrowserPredicate(t, ctx, browser, `(async () => { rejectCapabilityManifest(new Error('obsolete manifest failure')); await pendingCapabilityOpen; return location.hash === '#recipe' && !document.querySelector('#panel-chat').textContent.includes('obsolete manifest failure'); })()`)
}
