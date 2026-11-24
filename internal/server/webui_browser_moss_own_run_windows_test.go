//go:build windows

package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMOSSReloadFollowsItsOwnRun holds SQA-486's cure, which landed in b1d6e237a. Two draws of MOSS run
// at once from one browser: another tab's, then this tab's. Reloaded, the tab follows its own draw: its Cancel ends
// that draw, and the other runs on. A reloaded tab had followed the first live draw of its models, so its Cancel ended
// the other tab's (as it does against 6357cf23e's workflow.js).
func TestWebUIBrowserMOSSReloadFollowsItsOwnRun(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the own-run leg uses Chromium through cmd/webui-lane")
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	fixture := newMOSSGenerationFixture(t, func(ctx context.Context, _ artifact.Repository, _ modelrecipe.CapabilityEvidenceSelection, _ latentimage.Request) error {
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	registered := serveTestRequest(fixture.handler, http.MethodPost, "/library/register", `{"kind":"model","path":`+string(testcheck.Must(json.Marshal(fixture.root))(t))+`}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("register: %d %s", registered.Code, registered.Body)
	}
	var capabilities []struct{ Recipe artifact.ID }
	testcheck.NoError(t, json.Unmarshal(serveTestRequest(fixture.handler, http.MethodGet, "/generation/capabilities", "").Body.Bytes(), &capabilities))
	if len(capabilities) != 1 {
		t.Fatalf("the workspace lists %d capabilities; want MOSS's one", len(capabilities))
	}
	server := httptest.NewServer(fixture.handler)
	defer server.Close()
	defer fixture.handler.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	cancelShown := `(() => { const cancel=[...document.querySelectorAll('#panel-generation button')].find(button=>button.textContent==='Cancel'); return !!cancel && !cancel.hidden && !cancel.disabled; })()`
	settle(`!!document.querySelector('#panel-generation.active .workflow-task + .control-grid > label.control')`)
	// Another tab's draw, held, submitted first from the same browser, as a second tab open beside this one submits it.
	submission := string(testcheck.Must(json.Marshal(map[string]any{"task": recipe.TaskImageGen, "recipe": capabilities[0].Recipe, "input": map[string]any{"prompt": "the other tab's", "width": 32, "height": 48, "steps": 3}}))(t))
	check(`(() => { window.overgo.api.submit('/generation/run', ` + submission + `).then((accepted)=>{ window.otherOperation=accepted.operation; }); return true; })()`)
	settle(`!!window.otherOperation`)
	var other struct{ Operation artifact.ID }
	var otherID string
	testcheck.NoError(t, browser.Evaluate(ctx, `window.otherOperation`, &otherID))
	other.Operation = testcheck.Must(artifact.ParseID(otherID))(t)
	// This tab's draw, held beside it.
	check(`(() => {
  const panel=document.querySelector('#panel-generation');
  const prompt=[...panel.querySelectorAll('label.control')].find(label=>label.querySelector('span').textContent==='prompt').querySelector('textarea');
  prompt.value='this tab\'s'; prompt.dispatchEvent(new Event('input',{bubbles:true}));
  [...panel.querySelectorAll('button')].find(button=>button.textContent==='Run').click();
  return true;
})()`)
	settle(cancelShown)
	var statuses []struct {
		ID     artifact.ID
		Recipe artifact.ID
		State  string
	}
	testcheck.NoError(t, json.Unmarshal(serveTestRequest(fixture.handler, http.MethodGet, "/operations", "").Body.Bytes(), &statuses))
	own := artifact.ID{}
	for _, status := range statuses {
		if status.Recipe == capabilities[0].Recipe && status.ID != other.Operation {
			own = status.ID
		}
	}
	if !own.Valid() {
		t.Fatalf("the operations %+v hold no draw of this tab beside the other's %s", statuses, other.Operation)
	}
	// Reloaded, the tab follows a draw; its Cancel ends the one it follows.
	testcheck.NoError(t, browser.Reload(ctx))
	settle(cancelShown)
	check(`(() => { [...document.querySelectorAll('#panel-generation button')].find(button=>button.textContent==='Cancel').click(); return true; })()`)
	settle(`!![...document.querySelectorAll('#panel-generation button')].find(button=>button.textContent==='Run' && !button.disabled)`)
	state := func(id artifact.ID) string {
		var status struct{ State string }
		testcheck.NoError(t, json.Unmarshal(serveTestRequest(fixture.handler, http.MethodGet, "/operations?id="+id.String(), "").Body.Bytes(), &status))
		return status.State
	}
	if ownState, otherState := state(own), state(other.Operation); ownState != "cancelled" || otherState != "running" {
		t.Fatalf("the reloaded tab's Cancel left its own draw %s and the other tab's %s; want its own cancelled and the other running", ownState, otherState)
	}
	check(`window.overgo.errors.length===0`)
	webuilane.Leg(t, "moss-own-run", "A reloaded Generate tab follows the draw it submitted: its Cancel ends that draw while another tab's of the same model runs on.")
}
