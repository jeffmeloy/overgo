package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserInformationArchitecture drives where things live: the
// Library names each way a model or dataset comes in as its own section,
// each form under its heading; the Datasets tab with no catalog says so in
// plain words and its button opens the Library's dataset import with focus
// on its file field; Settings titles its Conversation and Connection parts.
func TestWebUIBrowserInformationArchitecture(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the information architecture leg uses Chromium through cmd/webui-lane")
	server := httptest.NewServer(newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{})))
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#library")
	defer browser.Close()
	// heading: the section title a field sits under.
	const heading = `(field) => { let node = field.closest('.row'); while (node && !node.classList.contains('section-title') && !node.querySelector?.(':scope > .section-title')) node = node.previousElementSibling; return node ? (node.classList.contains('section-title') ? node : node.querySelector(':scope > .section-title')).textContent : ''; }`

	// The Library: each way in is its own section, each form under its heading.
	settle(`!!document.querySelector('#panel-library.active input[aria-label="Chat dataset file on disk"]')`)
	check(`(() => {
  const titles = [...document.querySelectorAll('#panel-library .section-title')].map(title => title.textContent);
  const order = ['Hugging Face', 'Add a model from disk', 'Import a chat dataset', 'Downloads', 'Hosted providers'];
  const at = order.map(title => titles.indexOf(title));
  const under = ` + heading + `;
  return at.every((position, index) => position >= 0 && (index === 0 || position > at[index - 1]))
    && under(document.querySelector('#panel-library input[placeholder="model GGUF or directory on disk"]')) === 'Add a model from disk'
    && under(document.querySelector('#panel-library input[aria-label="Chat dataset file on disk"]')) === 'Import a chat dataset';
})()`)

	// Datasets with no catalog: plain words, and a way to the import.
	check(`(() => { location.hash = 'datasets'; return true; })()`)
	settle(`[...document.querySelectorAll('#panel-datasets.active button')].some(button => button.textContent === 'Import a chat dataset')`)
	check(`(() => { const text = document.querySelector('#panel-datasets').textContent; return text.includes('No datasets are catalogued yet.') && !/go run|cmd\//.test(text); })()`)
	check(`(() => { [...document.querySelectorAll('#panel-datasets button')].find(button => button.textContent === 'Import a chat dataset').click(); return true; })()`)
	settle(`location.hash === '#library' && document.activeElement && document.activeElement.getAttribute('aria-label') === 'Chat dataset file on disk'`)

	// Settings titles its conversation and connection parts; the conversation part is the chat's.
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "library-sections")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}
	check(`(() => { location.hash = 'chat'; return true; })()`)
	settle(`!!document.querySelector('#conversation-settings h3')`)
	check(`[...document.querySelectorAll('#settings-dialog h3')].map(title => title.textContent).join(',').includes('Conversation') && [...document.querySelectorAll('#settings-dialog h3')].some(title => title.textContent === 'Connection')`)
	webuilane.Leg(t, "information architecture leg", "the Library named each way a model or dataset comes in as its own section with each form under its heading; Datasets with no catalog said so plainly and its button opened the Library's dataset import with focus on its file field; Settings titled its Conversation and Connection parts")
}
