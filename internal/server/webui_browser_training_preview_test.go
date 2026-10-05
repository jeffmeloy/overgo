package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserTrainingPreview drives the training page's preview over a
// DPO workspace: with a dataset chosen, "Preview as trained" shows the first
// record's chosen and rejected sequences with the masked prompt prefix and
// the trained response marked token by token, pages to the next record, and
// draws the token lengths of every sequence.
func TestWebUIBrowserTrainingPreview(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the training preview leg uses Chromium through cmd/webui-lane")
	fixture := newDPOTrainingFixture(t)
	handler := newTestHandlerForRepository(t, fixture.store, &fakeGenerator{}, func(config *Config) { config.Workflows = fixture.workspace })
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option') && [...document.querySelectorAll('#panel-training-jobs button')].some(button => button.textContent === 'Preview as trained')`)
	check(`(() => {
  window.previewPanel = () => document.querySelector('#panel-training-jobs');
  window.previewButton = (label) => [...previewPanel().querySelectorAll('button')].find(button => button.textContent === label);
  window.previewStatus = () => previewButton('Preview as trained').parentElement.querySelector('[role="status"]').textContent;
  window.previewSequence = (label) => previewPanel().querySelector('[data-sequence="' + label + '"]');
  window.previewTokens = (label, kind) => [...previewSequence(label).querySelectorAll('.token-run span.' + kind)].map(token => token.textContent).join('');
  const dataset = [...previewPanel().querySelectorAll('label.control')].find(control => control.firstChild.textContent === 'dataset').querySelector('input');
  dataset.value = ` + "`" + fixture.dataset.String() + "`" + `; dataset.dispatchEvent(new Event('input', { bubbles: true }));
  previewButton('Preview as trained').click();
  return true;
})()`)

	// The first record: the prompt masked, each response trained, token by token.
	settle(`previewStatus() === 'record 1 of 2' && !!previewSequence('chosen') && !!previewSequence('rejected')`)
	check(`previewTokens('chosen', 'masked') === 'ab' && previewTokens('chosen', 'trained') === 'c' && previewTokens('rejected', 'masked') === 'ab' && previewTokens('rejected', 'trained') === 'd'`)
	check(`previewSequence('chosen').textContent.startsWith('chosen · 1 trained of 3 tokens') && previewButton('Previous').disabled && !previewButton('Next').disabled`)
	check(`[...previewPanel().querySelectorAll('[aria-label="Token lengths"] .pbar-row')].length > 0 && previewPanel().querySelector('[aria-label="Token lengths"]').textContent.includes('4 sequences')`)

	// The next record pages in.
	check(`(() => { previewButton('Next').click(); return true; })()`)
	settle(`previewStatus() === 'record 2 of 2' && previewTokens('chosen', 'masked') === 'ba' && previewTokens('chosen', 'trained') === 'dc' && previewButton('Next').disabled && !previewButton('Previous').disabled`)
	webuilane.Leg(t, "training preview leg", "the preview showed each record's prompt masked and responses trained token by token, paged between records, and drew the token lengths of every sequence")
}
