package server

import (
	"fmt"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserSFTControls drives a conversation fine-tune on the Train
// tab: the dataset is chosen by name from a picker holding the registered
// conversation datasets, the page says how a run learns from them, previewing
// shows each conversation as rendered with every token trained, and a finished
// run shows the held-out verdict, base against trained, beside the trained
// model; after a reload the choice is still the one made.
func TestWebUIBrowserSFTControls(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the conversation fine-tuning leg uses Chromium through cmd/webui-lane")
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
  window.verdict = () => trainingPanel().querySelector('[aria-label="Held-out verdict"]');
  return true;
})()`
	id := fixture.conversations.String()
	open := func() {
		t.Helper()
		check(`(() => { location.hash = 'training-jobs'; return true; })()`)
		settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
		check(helpers)
	}
	open()

	// The dataset is a picker of names: the conversation dataset by name, its identity the value.
	settle(fmt.Sprintf(`!!control('conversations') && [...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q && option.text === 'chats · 12 files')`, id))
	// The page says where the run happens and how it learns, beside the controls.
	check(`/Runs on the host processor/.test(trainingPanel().querySelector('.workflow-placement').textContent) && /Learns every token of each conversation/.test(trainingPanel().querySelector('.workflow-placement').textContent)`)
	check(fmt.Sprintf(`(() => {
  const select = control('conversations').querySelector('select');
  select.value = %q;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  control('steps').querySelector('input').value = '4';
  control('output').querySelector('input').value = 'tuned';
  return select.value === %q;
})()`, id, id))

	// Previewed, a conversation reads as the template rendered it, every token trained.
	check(`(() => { press('Preview as trained'); return true; })()`)
	settle(`!!trainingPanel().querySelector('[data-sequence="conversation"]')`)
	check(`(() => {
  const sequence = trainingPanel().querySelector('[data-sequence="conversation"]');
  const trained = sequence.querySelectorAll('.token-run .trained').length;
  return trained > 0 && sequence.querySelectorAll('.token-run .masked').length === 0 && sequence.querySelector('.note').textContent === 'conversation · ' + trained + ' trained of ' + trained + ' tokens';
})()`)

	// A finished run shows the held-out verdict beside the trained model and the run.
	check(`(() => { press('Run'); return true; })()`)
	settle(`!!verdict()`)
	check(`(() => {
  const values = Object.fromEntries([...verdict().querySelectorAll('.stat')].map((stat) => [stat.querySelector('.k').textContent, stat.querySelector('.v').textContent]));
  return /^[0-9]\.[0-9]{3}$/.test(values['Base model']) && /^[0-9]\.[0-9]{3}$/.test(values['Trained model']) && ['beats base', 'below base', 'level with base'].includes(values['Verdict']) &&
    /trained model/.test(verdict().textContent) && /verdict/.test(verdict().textContent) && /run/.test(verdict().textContent);
})()`)

	// Reloaded, the picker still holds the dataset chosen, though its name arrives after the choice is restored.
	if err := browser.Reload(ctx); err != nil {
		t.Fatal(err)
	}
	open()
	settle(fmt.Sprintf(`!!control('conversations') && control('conversations').querySelector('select').value === %q && control('conversations').querySelector('select').selectedOptions[0].text === 'chats · 12 files' && control('steps').querySelector('input').value === '4'`, id))
}
