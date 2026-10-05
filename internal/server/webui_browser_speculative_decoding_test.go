package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserSpeculativeDecoding drives speculative decoding from
// Settings: a served model declaring a draft head offers the toggle with its
// draft span, a turn with it on asks the server to speculate at temperature
// 0, and the turn's inspector shows the drafts accepted; a model without a
// draft head offers no toggle.
func TestWebUIBrowserSpeculativeDecoding(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the speculative decoding leg uses Chromium through cmd/webui-lane")
	drafting := httptest.NewServer(newTestHandlerWithRepository(t, &draftGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{})}))
	defer drafting.Close()
	plain := httptest.NewServer(newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{})))
	defer plain.Close()
	ctx, browser, check, settle := openBrowserLeg(t, drafting.URL)
	defer browser.Close()
	const toggle = `document.querySelector('#conversation-settings input[aria-label="speculative decoding"]')`
	settle(`!!document.querySelector('.composer textarea') && !!document.querySelector('#conversation-settings input[aria-label="temperature"]')`)

	// The toggle names the declared draft span; on, a turn speculates at temperature 0.
	check(`!!` + toggle + ` && ` + toggle + `.parentElement.textContent.includes('draft 3 tokens a step')`)
	check(`(() => {
  const box = ` + toggle + `; box.checked = true; box.dispatchEvent(new Event('change', { bubbles: true }));
  window.speculativeStream = overgo.api.stream; window.speculativeRequests = [];
  overgo.api.stream = async function (path, body, options) { if (path === '/v1/responses') speculativeRequests.push(body); return speculativeStream.call(this, path, body, options); };
  const input = document.querySelector('.composer textarea');
  input.value = 'draft a reply'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.chat-log .msg.assistant')].some(message => message.textContent.includes('drafted reply') && [...message.querySelectorAll('button')].some(button => button.textContent === 'inspect'))`)
	check(`speculativeRequests.length === 1 && speculativeRequests[0].speculative === true && speculativeRequests[0].temperature === 0`)

	// The turn's inspector shows the drafts accepted.
	check(`(() => { [...document.querySelectorAll('.chat-log .msg.assistant button')].find(button => button.textContent === 'inspect').click(); return true; })()`)
	settle(`[...document.querySelectorAll('#inspector .stat')].some(card => card.textContent.includes('Drafts accepted') && card.textContent.includes('5 / 6') && card.textContent.includes('83% speculative'))`)

	// A model without a draft head offers no toggle.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+plain.URL+"`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + plain.URL + "`" + ` && !!document.querySelector('#conversation-settings input[aria-label="temperature"]')`)
	check(`!` + toggle)
	webuilane.Leg(t, "speculative decoding leg", "a served model declaring a draft head offered the toggle with its draft span, a turn with it on asked to speculate at temperature 0 and its inspector showed the drafts accepted, and a model without a draft head offered no toggle")
}
