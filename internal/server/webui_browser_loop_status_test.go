package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserLoopStatus drives the unattended loop on the Runtime tab:
// it states that the loop runs, the row it works and its last landing; Stop
// records a user stop from the workbench and the tab shows the loop stopped
// with its reason and how to resume; a server with no loop shows none.
func TestWebUIBrowserLoopStatus(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the loop status leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	loop := &fakeLoop{status: runningLoop()}
	looped := httptest.NewServer(newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) { config.Loop = loop }))
	defer looped.Close()
	plain := httptest.NewServer(newTestHandlerWithRepository(t, &fakeGenerator{}))
	defer plain.Close()
	ctx, browser, check, settle := openBrowserLeg(t, looped.URL+"/#runtime")
	defer browser.Close()

	// The loop runs, working a row after its last landing.
	settle(`(() => { const loop = document.querySelector('#panels .loop-status'); return !!loop && !loop.hidden && !!loop.querySelector('.loop-state'); })()`)
	check(`(() => {
  const loop = document.querySelector('#panels .loop-status');
  return loop.querySelector('.loop-state').textContent === 'running' && loop.querySelector('.loop-row').textContent === 'Works gui-next/do: Build the next view'
    && loop.querySelector('.loop-landing').textContent.startsWith('Last landing: Show a turn\'s tokens per forward (gui-phase-profile/do, 0f8480fcacd5, ')
    && !loop.querySelector('.loop-stop') && !loop.querySelector('button').disabled && !loop.textContent.includes('null');
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "loop-status")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// Stop records a user stop; the tab shows the loop stopped and how to resume.
	check(`(() => { document.querySelector('#panels .loop-status button').click(); return true; })()`)
	settle(`document.querySelector('#panels .loop-status .loop-state').textContent === 'stopped'`)
	check(`(() => {
  const loop = document.querySelector('#panels .loop-status');
  return loop.querySelector('.loop-stop').textContent === 'Stopped (user-stop: stopped from the workbench). Resume from the checkout with plan -resume-stop.'
    && !loop.querySelector('button') && loop.querySelector('.loop-row').textContent === 'Works gui-next/do: Build the next view';
})()`)
	loop.mu.Lock()
	recorded := append([]string(nil), loop.details...)
	loop.mu.Unlock()
	if len(recorded) != 1 || recorded[0] != loopStopDetail {
		t.Fatalf("the stop recorded %v", recorded)
	}

	// A server with no loop shows none.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+plain.URL+"/#runtime`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + plain.URL + "`" + ` && !!document.querySelector('#panels table.grid')`)
	check(`!document.querySelector('#panels .loop-status') || document.querySelector('#panels .loop-status').hidden`)
	webuilane.Leg(t, "loop status leg", "the Runtime tab stated that the loop runs, the row it works and its last landing; Stop recorded a user stop from the workbench and the tab showed the loop stopped with its reason and how to resume; a server with no loop showed none")
}
