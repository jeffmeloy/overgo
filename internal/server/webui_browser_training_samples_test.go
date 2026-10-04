package server

import (
	"fmt"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserTrainingSamples drives a conversation fine-tune that samples
// every fourth update on the Train tab: the page offers the interval as a
// field, left empty to sample nothing; a finished run shows, beside its
// held-out verdict, the samples it generated after each held-out prompt, the
// prompt and the reply the conversation holds above them, in step order from
// before training to the last update.
func TestWebUIBrowserTrainingSamples(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the training samples leg uses Chromium through cmd/webui-lane")
	}
	fixture := newSFTFixture(t)
	handler := newTestHandlerForRepository(t, fixture.store, &fakeGenerator{}, func(config *Config) { config.Workflows = fixture.workspace })
	server := httptest.NewServer(handler)
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
	const helpers = `(() => {
  window.trainingPanel = () => document.querySelector('#panel-training-jobs');
  window.control = (label) => [...trainingPanel().querySelectorAll('label.control')].find((item) => item.querySelector('span').textContent === label);
  window.press = (text) => [...trainingPanel().querySelectorAll('button')].find((button) => button.textContent === text).click();
  return true;
})()`
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
	check(helpers)

	// The interval is a number field beside the other choices, empty until set.
	settle(fmt.Sprintf(`!!control('sample every') && control('sample every').querySelector('input').value === '' && [...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q)`, fixture.conversations.String()))
	check(fmt.Sprintf(`(() => {
  const select = control('conversations').querySelector('select');
  select.value = %q;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  control('steps').querySelector('input').value = '12';
  control('sample every').querySelector('input').value = '4';
  control('output').querySelector('input').value = 'sampled';
  return select.value === %q;
})()`, fixture.conversations.String(), fixture.conversations.String()))
	check(`(() => { press('Run'); return true; })()`)

	// The samples arrive with the verdict, in step order.
	settle(`!!trainingPanel().querySelector('[aria-label="Training samples"]') && !!trainingPanel().querySelector('[aria-label="Held-out verdict"]')`)
	check(`(() => {
  const strips = [...trainingPanel().querySelectorAll('[aria-label="Training samples"] [data-record]')];
  if (strips.length === 0) return false;
  return strips.every((strip) => {
    const steps = [...strip.querySelectorAll('li')].map((item) => item.dataset.step).join(',');
    const prompts = [...strip.querySelectorAll('.preview-text')];
    return steps === '0,4,8,12' && prompts.length >= 2 && prompts[0].textContent !== '' && prompts[1].textContent !== '' &&
      [...strip.querySelectorAll('li .preview-text')].every((text) => text.textContent !== '');
  });
})()`)
}
