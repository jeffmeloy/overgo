package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserChatDefaults drives the chat's output budget and its
// output limit: Settings starts at the output the server accepts, not the
// API's default budget, and a reply cut at the limit set there says so and
// offers Continue, which carries the reply on in the same message.
func TestWebUIBrowserChatDefaults(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the chat defaults leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &sessionGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), blocked: make(chan struct{})}
	server := httptest.NewServer(newTestHandlerForRepository(t, store, generator, func(config *Config) { config.MaxTokens = 48 }))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	const (
		budget         = `document.querySelector('#conversation-settings input[aria-label="max tokens"]')`
		reply          = `document.querySelector('.chat-log .msg.assistant .body').textContent.replace('▍', '').trim()`
		continueButton = `[...document.querySelectorAll('.chat-log .note button')].find(button => button.textContent === 'Continue')`
	)

	// Settings starts at the output the server accepts, above the API's default budget.
	settle(`!!document.querySelector('.composer textarea') && !!` + budget)
	check(budget + `.value === '48' && overgo.capabilities().generation.max_tokens < 48`)

	// A reply cut at its limit says so and offers Continue, which carries it on in the same message.
	check(`(() => {
  const limit = ` + budget + `; limit.value = '3'; limit.dispatchEvent(new Event('input', { bubbles: true }));
  const input = document.querySelector('.composer textarea');
  input.value = 'say the sentence'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.chat-log .note')].some(note => note.textContent.startsWith('Output limit reached.')) && !!` + continueButton)
	check(reply + ` === 'The quick brown'`)
	check(`(() => { ` + continueButton + `.click(); return true; })()`)
	settle(`document.querySelectorAll('.chat-log .msg.assistant').length === 1 && ` + reply + ` === 'The quick brown fox jumps over'`)
	webuilane.Leg(t, "chat defaults leg", "Settings started at the output the server accepts rather than the API's default budget, and a reply cut at its output limit said so and offered Continue, which carried the reply on in the same message")
}
