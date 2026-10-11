package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserTopPSetting holds the chat's top-p setting. Settings shows
// a top-p field beside temperature and the output limit, empty, with the
// server's own default as its placeholder, and what a person types there is
// kept across a reload. A turn sent with the field empty names no top-p, and
// the turn's record shows the server's default; a turn sent with it set
// names it, and the record shows it; and the same turn continued names it
// again, since a continuation asked under other sampling than its turn's is
// refused. The three fields fit their dialog at desktop and phone widths in
// both colour schemes.
func TestWebUIBrowserTopPSetting(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the top-p setting leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &sessionGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), blocked: make(chan struct{})}
	server := httptest.NewServer(newTestHandlerForRepository(t, store, generator, func(config *Config) { config.MaxTokens = 48 }))
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	const (
		field  = `document.querySelector('#conversation-settings input[aria-label="top-p"]')`
		heat   = `document.querySelector('#conversation-settings input[aria-label="temperature"]')`
		budget = `document.querySelector('#conversation-settings input[aria-label="max tokens"]')`
		reply  = `[...document.querySelectorAll('.chat-log .msg.assistant .body')].at(-1).textContent.replace('▍', '').trim()`
		more   = `[...document.querySelectorAll('.chat-log .note button')].findLast(button => button.textContent === 'Continue')`
		failed = `document.querySelector('.chat-log .note[role="alert"]')`
		posted = `window.topPRequests.at(-1)`
		// recorded is what the server's record of the latest turn says it sampled with.
		recorded = `(await overgo.api.get('/interactions/inspect?response=' + encodeURIComponent(overgo.conversation().latest))).sampling`
	)
	// typed puts a value in a settings field as a person's typing does.
	typed := func(target, value string) string {
		return `(() => { const input = ` + target + `; input.value = '` + value + `'; input.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`
	}
	// sent types a message and sends it once the page will take one.
	sent := func(text string) {
		t.Helper()
		check(typed(`document.querySelector('.composer textarea')`, text))
		settle(`!document.querySelector('.send-button').disabled`)
		check(`(() => { document.querySelector('.send-button').click(); return true; })()`)
	}

	// The field: empty, showing the default an empty field leaves standing, beside the other two.
	settle(`!!document.querySelector('.composer textarea') && !!` + heat)
	check(`!!` + field + ` && ` + field + `.value === '' && ` + field + `.placeholder === String(overgo.capabilities().generation.top_p) && ` +
		field + `.closest('.settings-fields') === ` + heat + `.closest('.settings-fields') && ` + field + `.closest('.settings-fields') === ` + budget + `.closest('.settings-fields')`)

	// What is typed there is kept across a reload.
	check(typed(field, "0.5"))
	if err := browser.Reload(ctx); err != nil {
		t.Fatal(err)
	}
	settle(`!!` + field + ` && ` + field + `.value === '0.5'`)
	check(typed(field, ""))

	// Empty: the turn names no top-p, and the server's default stands.
	check(`(() => {
  window.topPStream = overgo.api.stream; window.topPRequests = [];
  overgo.api.stream = async function (path, body, options) { if (path === '/v1/responses') topPRequests.push(body); return topPStream.call(this, path, body, options); };
  return true;
})()`)
	sent("say the sentence")
	settle(`(` + reply + ` === 'The quick brown fox jumps over the dog.' && !!overgo.conversation() && !!overgo.conversation().latest) || !!` + failed)
	check(`(async () => {
  if (` + failed + `) throw new Error('the turn failed: ' + ` + failed + `.textContent);
  const sampling = ` + recorded + `;
  if (Object.hasOwn(` + posted + `, 'top_p') || !sampling || sampling.top_p !== overgo.capabilities().generation.top_p) throw new Error('with the field empty the turn posted ' + JSON.stringify(` + posted + `) + ' and was recorded as sampled with ' + JSON.stringify(sampling) + ', want no top_p posted and the server default recorded');
  return true;
})()`)

	// Set: the turn names it, and its record shows it.
	check(typed(field, "0.5"))
	check(typed(budget, "3"))
	sent("say the sentence again")
	settle(`([...document.querySelectorAll('.chat-log .note')].some(note => note.textContent.startsWith('Output limit reached.')) && !!` + more + `) || !!` + failed)
	check(`(async () => {
  if (` + failed + `) throw new Error('the turn failed: ' + ` + failed + `.textContent);
  const sampling = ` + recorded + `;
  if (` + posted + `.top_p !== 0.5 || !sampling || sampling.top_p !== 0.5) throw new Error('with top-p set to 0.5 the turn posted ' + JSON.stringify(` + posted + `) + ' and was recorded as sampled with ' + JSON.stringify(sampling));
  return true;
})()`)

	// Continued: the continuation names the same top-p, so it is not refused as asked under other sampling.
	check(`(() => { ` + more + `.click(); return true; })()`)
	settle(reply + ` === 'The quick brown fox jumps over' || !!` + failed)
	check(`(() => {
  if (` + failed + `) throw new Error('the turn continued failed: ' + ` + failed + `.textContent);
  if (` + posted + `.continue !== true || ` + posted + `.top_p !== 0.5) throw new Error('the continuation posted ' + JSON.stringify(` + posted + `) + ', want it to continue under top_p 0.5');
  return true;
})()`)
	webuilane.Leg(t, "top-p setting leg", "Settings showed an empty top-p field with the server's default as its placeholder, kept a typed value across a reload, and a turn named top-p only when it was set, as did the same turn continued")

	// The three fields fit their dialog at each width, in each colour scheme.
	for _, scheme := range webuilane.ColourSchemes {
		if err := browser.SetColorScheme(ctx, scheme); err != nil {
			t.Fatal(err)
		}
		for _, viewport := range webuilane.ScreenViewports {
			if err := webuilane.SettleViewport(ctx, browser, viewport); err != nil {
				t.Fatal(err)
			}
			check(`(() => {
  const dialog = document.querySelector('#settings-dialog');
  if (!dialog.open) document.querySelector('#settings-toggle').click();
  const within = dialog.getBoundingClientRect();
  for (const input of [` + heat + `, ` + field + `, ` + budget + `]) {
    const box = input.getBoundingClientRect();
    if (!(box.width > 0) || box.left < within.left || box.right > within.right) throw new Error('at ` + viewport.Name + ` in the ` + scheme + ` scheme the ' + input.getAttribute('aria-label') + ' field spans ' + box.left + ' to ' + box.right + ' in a dialog spanning ' + within.left + ' to ' + within.right);
  }
  if (dialog.scrollWidth > dialog.clientWidth) throw new Error('at ` + viewport.Name + ` in the ` + scheme + ` scheme Settings scrolls sideways');
  return true;
})()`)
			if dir := os.Getenv("OVERGO_WEBUI_LANE_SCREENS"); dir != "" {
				if _, err := webuilane.CaptureState(ctx, browser, dir, viewport, "settings-top-p-"+scheme); err != nil {
					t.Fatal(err)
				}
			}
			check(`(() => { const dialog = document.querySelector('#settings-dialog'); if (dialog.open) dialog.close(); return true; })()`)
		}
	}
}
