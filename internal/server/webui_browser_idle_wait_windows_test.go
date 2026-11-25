//go:build windows

package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserAWaitThatCannotComeTrueFailsWhenIdle holds idle-wait/do (the owner's review of 2026-11-24, cause 3):
// a browser wait whose predicate cannot come true fails once the page is idle, saying what the page shows, though the
// page holds the activity stream open and its intervals poll; and a wait on a run the server is still drawing is not
// called idle, the page awaiting the run it follows. The waits had looped until their predicates held: four hangs on
// 2026-11-24, one a full lane stalled 20 minutes.
func TestWebUIBrowserAWaitThatCannotComeTrueFailsWhenIdle(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the idle-wait leg uses Chromium through cmd/webui-lane")
	release := make(chan struct{})
	free := sync.OnceFunc(func() { close(release) })
	fixture := newMOSSGenerationFixture(t, func(ctx context.Context, _ artifact.Repository, _ modelrecipe.CapabilityEvidenceSelection, _ latentimage.Request) error {
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	t.Cleanup(free)
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	// The held run is released when the page's own wait asks, many frames into it.
	routes := http.NewServeMux()
	routes.HandleFunc("/test/release", func(response http.ResponseWriter, _ *http.Request) {
		free()
		response.WriteHeader(http.StatusNoContent)
	})
	routes.Handle("/", fixture.handler)
	server := httptest.NewServer(routes)
	defer server.Close()
	defer fixture.handler.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active .workflow-task + .control-grid > label.control')`)
	check(`!!window.__laneIdle`)
	// A predicate nothing will make true ends the wait once the page is idle, with what it shows.
	err := browser.Eventually(ctx, `document.title === 'a title this page never has'`)
	if err == nil || !strings.Contains(err.Error(), "the page is idle") || !strings.Contains(err.Error(), "Generate") {
		t.Fatalf("a wait that cannot come true ended %v; want it failed as idle, with the page's panels", err)
	}
	// A run the server is drawing keeps the page from idle: its picture is awaited until the run, released many frames
	// into the wait, draws it.
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea');
  prompt.value='a lighthouse'; prompt.dispatchEvent(new Event('input',{bubbles:true}));
  [...panel.querySelectorAll('button')].find(button=>button.textContent==='Run').click();
  return true;
})()`)
	settle(`(() => { window.idleFrames=(window.idleFrames||0)+1; if (window.idleFrames===120) fetch('/test/release'); return !!document.querySelector('#panel-generation .workbench-output'); })()`)
	check(`window.overgo.errors.length===0 || (() => { throw new Error('the page recorded errors: '+JSON.stringify(window.overgo.errors).slice(0,600)); })()`)
	webuilane.Leg(t, "idle-wait", "A browser wait that cannot come true fails once the page is idle, saying what it shows, and a wait on a run the server is drawing is not called idle.")
}
