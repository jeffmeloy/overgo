package server

import (
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserReasoningLevel drives the reasoning level from Settings on
// a model whose chat template reads one: the control offers the model
// default, off and the levels that change the prompt, and a turn's chosen
// level reaches the template; a model whose template reads none offers no
// control.
func TestWebUIBrowserReasoningLevel(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the reasoning level leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recorder := &promptRecorder{Runner: openTemplateRunner(t, 64, reasoningTemplate)}
	reasoned := httptest.NewServer(newTestHandlerForRepository(t, store, recorder))
	defer reasoned.Close()
	plain := httptest.NewServer(newTestHandler(t, openSessionRunner(t, 64)))
	defer plain.Close()
	ctx, browser, check, settle := openBrowserLeg(t, reasoned.URL)
	defer browser.Close()
	const control = `document.querySelector('#conversation-settings select[aria-label="reasoning level"]')`
	lastPrompt := func() string {
		recorder.mu.Lock()
		defer recorder.mu.Unlock()
		if len(recorder.prompts) == 0 {
			return ""
		}
		return recorder.prompts[len(recorder.prompts)-1]
	}
	settle(`!!document.querySelector('.composer textarea') && !!` + control)
	check(`[...` + control + `.options].map(option => option.value).join() === ',off,low,medium,high'`)
	send := func(level string, turns int) {
		t.Helper()
		check(`(() => {
  const control = ` + control + `; control.value = '` + level + `'; control.dispatchEvent(new Event('change', { bubbles: true }));
  const input = document.querySelector('.composer textarea');
  input.value = 'ab'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
		settle(`document.querySelectorAll('.chat-log .msg.assistant').length === ` + strconv.Itoa(turns) + ` && !document.querySelector('.send-button').disabled`)
	}

	// A chosen level reaches the template; off leaves thinking out.
	send("high", 1)
	if prompt := lastPrompt(); !strings.HasPrefix(prompt, "dcc") {
		t.Fatalf("a high turn's prompt = %q", prompt)
	}
	send("off", 2)
	if prompt := lastPrompt(); strings.HasPrefix(prompt, "d") {
		t.Fatalf("an off turn's prompt = %q", prompt)
	}

	// A template reading no reasoning offers no control.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+plain.URL+"`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + plain.URL + "`" + ` && !!document.querySelector('#conversation-settings input[aria-label="temperature"]')`)
	check(`!` + control)
	webuilane.Leg(t, "reasoning level leg", "a model whose chat template reads a reasoning level offered the model default, off and the levels that change its prompt, and a turn's chosen level reached the template; a template reading none offered no control")
}
