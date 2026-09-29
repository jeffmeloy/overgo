package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserEmptyStates drives the states a page shows before it has
// anything to show: the Agent tab with no agents says so and opens the
// creation fold, its chat waiting for an agent; the Library's search and
// download tables stay hidden until they have rows; one page of runs shows
// no pager; a long conversation
// title wraps with the whole title in its tooltip; and the memory bar says
// the model's share of the device in words.
func TestWebUIBrowserEmptyStates(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the empty states leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	server := httptest.NewServer(newTestHandlerForRepository(t, store, screensGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), path: writeTensorFixture(t)}))
	defer server.Close()
	measured := httptest.NewServer(newTestHandler(t, &memoryOwnersGenerator{fakeGenerator: &fakeGenerator{}, memory: testDeviceMemory()}))
	defer measured.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL+"/#agent")
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
	const visible = `(node) => !!node && node.getClientRects().length > 0`

	// Agent with no agents: a statement and a way to create one; chat waits for an agent.
	settle(`[...document.querySelectorAll('#panel-agent.active button')].some(button => button.textContent === 'Create an agent')`)
	check(`(() => {
  const panel = document.querySelector('#panel-agent'), shown = ` + visible + `;
  return panel.textContent.includes('No agents yet.') && ![...panel.querySelectorAll('button')].some(button => ['Pause', 'Resume'].includes(button.textContent) && shown(button))
    && shown(panel.querySelector('.agent-chat-note')) && !shown(panel.querySelector('.agent-chat-note + div textarea'));
})()`)
	check(`(() => { [...document.querySelectorAll('#panel-agent button')].find(button => button.textContent === 'Create an agent').click(); return true; })()`)
	settle(`[...document.querySelectorAll('#panel-agent details.fold')].some(fold => fold.querySelector('summary').textContent === 'New agent' && fold.open)`)

	// Library: search and download tables stay hidden until they have rows.
	check(`(() => { location.hash = 'library'; return true; })()`)
	settle(`!!document.querySelector('#panel-library.active input[aria-label="Chat dataset file on disk"]')`)
	check(`(() => { const shown = ` + visible + `; const tables = [...document.querySelectorAll('#panel-library table.hide-empty')].filter(table => table.querySelector('th').textContent.toLowerCase() === 'repository'); return tables.length === 2 && tables.every(table => !shown(table)); })()`)

	// Runs with none: a sentence, no pager, no header-only table.
	check(`(() => { location.hash = 'runs'; return true; })()`)
	settle(`!!document.querySelector('#panel-runs.active') && document.querySelector('#panel-runs').textContent.includes('No training runs yet.')`)
	check(`(() => { const shown = ` + visible + `; const buttons = [...document.querySelectorAll('#panel-runs button')].filter(button => ['prev', 'next'].includes(button.textContent)); return buttons.length === 2 && buttons.every(button => !shown(button)) && !document.querySelector('#panel-runs table'); })()`)

	// A long conversation title wraps, the whole title in its tooltip.
	const title = "compare the quarterly results across every regional office"
	check(`(() => { location.hash = 'chat'; return true; })()`)
	settle(`!!document.querySelector('#panel-chat.active .composer textarea')`)
	check(`(() => {
  const input = document.querySelector('.composer textarea');
  input.value = '` + title + `'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.conversation-open')].some(button => button.title.startsWith('` + title + `'))`)
	// The rail shows on the front view, in the navigation drawer at this width.
	check(`(() => { if (!document.querySelector('.shell').classList.contains('front')) document.getElementById('workbench-toggle').click(); return true; })()`)
	check(`(() => { if (!document.getElementById('navigation').getClientRects().length) document.getElementById('navigation-toggle').click(); return true; })()`)
	settle(`document.querySelector('.shell').classList.contains('front') && [...document.querySelectorAll('.conversation-open')].some(button => button.getClientRects().length > 0)`)
	check(`(() => {
  const button = [...document.querySelectorAll('.conversation-open')].find(candidate => candidate.title.startsWith('` + title + `'));
  const name = button.querySelector('.conversation-title'), turns = button.querySelector('.note');
  return getComputedStyle(name).webkitLineClamp === '2' && name.getBoundingClientRect().height > turns.getBoundingClientRect().height * 1.5;
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "empty-states-rail")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// The memory bar says the model's share of the device in words.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+measured.URL+"`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + measured.URL + "`" + ` && !document.getElementById('memory-label').hidden`)
	check(`document.getElementById('memory-label').textContent === '12.00 GB of 24.00 GB'`)
	webuilane.Leg(t, "empty states leg", "Agent with no agents said so and opened its creation fold, its chat waiting for an agent; the Library's search and download tables stayed hidden until they had rows; runs with none said so with no pager or empty table; a long conversation title wrapped with the whole title in its tooltip; the memory bar said the model's share of the device in words")
}
