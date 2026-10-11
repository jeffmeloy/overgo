package server

import (
	"fmt"
	"net/http/httptest"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserFineTuneLossCurve holds the Train tab to drawing a
// conversation fine-tune's loss. The run reports its loss after each update
// under the name loss, and the tab drew only series whose name ends in
// _loss, as a preference run's does, so a fine-tune's curve was carried to
// the page and never drawn. A fine-tune of four updates, finished, shows one
// curve named loss over its four steps.
func TestWebUIBrowserFineTuneLossCurve(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the fine-tune loss leg uses Chromium through cmd/webui-lane")
	fixture := newSFTFixture(t)
	handler := newTestHandlerForRepository(t, fixture.store, &fakeGenerator{}, func(config *Config) { config.Workflows = fixture.workspace })
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
	check(`(() => {
  window.trainingPanel = () => document.querySelector('#panel-training-jobs');
  window.control = (label) => [...trainingPanel().querySelectorAll('label.control')].find((item) => item.querySelector('span').textContent === label);
  return true;
})()`)
	settle(fmt.Sprintf(`!!control('conversations') && [...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q)`, fixture.conversations.String()))
	check(fmt.Sprintf(`(() => {
  const select = control('conversations').querySelector('select');
  select.value = %q;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  control('steps').querySelector('input').value = '4';
  control('output').querySelector('input').value = 'curve';
  [...trainingPanel().querySelectorAll('button')].find((button) => button.textContent === 'Run').click();
  return select.value === %q;
})()`, fixture.conversations.String(), fixture.conversations.String()))
	settle(`!!trainingPanel().querySelector('[aria-label="Held-out verdict"]') || /^failed/.test(trainingPanel().querySelector('[aria-label="run status"]').textContent)`)
	check(`(() => {
  const drawn = [...trainingPanel().querySelectorAll('[data-trend]')].map((trend) => trend.dataset.trend + ': ' + trend.textContent);
  const curve = trainingPanel().querySelector('[data-trend="loss"]');
  if (!curve || !curve.querySelector('svg') || !curve.textContent.startsWith('loss · 4 steps')) throw new Error('the finished fine-tune shows the curves ' + JSON.stringify(drawn) + ', want its loss over 4 steps; the run status reads ' + trainingPanel().querySelector('[aria-label="run status"]').textContent);
  return true;
})()`)
	webuilane.Leg(t, "fine-tune loss leg", "a conversation fine-tune of four updates showed its loss on the Train tab as one curve over its four steps")
}
