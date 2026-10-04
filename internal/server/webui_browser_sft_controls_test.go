package server

import (
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/trainingworkflow"
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
	check(`/Runs on the host processor/.test(trainingPanel().querySelector('.workflow-placement').textContent) && /Learns the replies of each conversation/.test(trainingPanel().querySelector('.workflow-placement').textContent)`)
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
  const trained = sequence.querySelectorAll('.token-run .trained').length, masked = sequence.querySelectorAll('.token-run .masked').length;
  return trained > 0 && masked > 0 && sequence.querySelector('.note').textContent === 'conversation · ' + trained + ' trained of ' + (trained + masked) + ' tokens';
})()`)
	// Trained on the prompts too, every token is learned; the choice is then put back.
	check(`(() => { const select = control('train on prompts').querySelector('select'); select.value = 'true'; select.dispatchEvent(new Event('input', { bubbles: true })); return select.value === 'true'; })()`)
	check(`(() => { press('Preview as trained'); return true; })()`)
	settle(`(() => { const sequence = trainingPanel().querySelector('[data-sequence="conversation"]'); return !!sequence && sequence.querySelectorAll('.token-run .masked').length === 0 && sequence.querySelectorAll('.token-run .trained').length > 0; })()`)
	check(`(() => { const select = control('train on prompts').querySelector('select'); select.value = ''; select.dispatchEvent(new Event('input', { bubbles: true })); return select.value === ''; })()`)

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

// TestWebUIBrowserLoRAControls drives a LoRA fine-tune on the Train tab: the
// capability list offers the adapter beside fine-tuning the whole model, and
// choosing it brings a rank field that starts at the default, a note that says
// the adapter is saved as PEFT, and, beside the conversations picker, the same
// preview; a finished run shows the held-out verdict as a whole-model run does.
func TestWebUIBrowserLoRAControls(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the LoRA fine-tuning leg uses Chromium through cmd/webui-lane")
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
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
	check(helpers)

	// Both fine-tunings are listed; the whole-model one is first and has no rank.
	check(`[...trainingPanel().querySelectorAll('select[aria-label="capability"] option')].map((option) => option.text).join('|') === 'training / served, on conversations|training / served, LoRA adapter on conversations' && !control('rank')`)
	// Choosing the adapter brings its rank, at the default, and says where the adapter goes.
	check(`(() => {
  const select = trainingPanel().querySelector('select[aria-label="capability"]');
  select.value = [...select.options].find((option) => /LoRA/.test(option.text)).value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	settle(`!!control('rank') && control('rank').querySelector('input').value === '8'`)
	check(`/Hugging Face PEFT adapter in the output's adapter directory/.test(trainingPanel().querySelector('.workflow-placement').textContent)`)
	settle(fmt.Sprintf(`[...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q)`, fixture.conversations.String()))
	check(fmt.Sprintf(`(() => {
  const select = control('conversations').querySelector('select');
  select.value = %q;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  control('steps').querySelector('input').value = '%d';
  control('rank').querySelector('input').value = '2';
  control('output').querySelector('input').value = 'adapted';
  return select.value === %q;
})()`, fixture.conversations.String(), adapterSteps, fixture.conversations.String()))

	// The conversations preview as rendered, as for the whole model.
	check(`(() => { press('Preview as trained'); return true; })()`)
	settle(`!!trainingPanel().querySelector('[data-sequence="conversation"]')`)

	// A finished run shows the held-out verdict, and its adapter is on disk as PEFT.
	check(`(() => { press('Run'); return true; })()`)
	settle(`!!verdict()`)
	check(`(() => {
  const values = Object.fromEntries([...verdict().querySelectorAll('.stat')].map((stat) => [stat.querySelector('.k').textContent, stat.querySelector('.v').textContent]));
  return /^[0-9]\.[0-9]{3}$/.test(values['Base model']) && /^[0-9]\.[0-9]{3}$/.test(values['Trained model']) && ['beats base', 'below base', 'level with base'].includes(values['Verdict']);
})()`)
	if _, err := os.Stat(filepath.Join(fixture.roots.Checkpoints, "adapted", trainingworkflow.LoRAAdapterDirectory, "adapter_config.json")); err != nil {
		t.Fatalf("the run left no PEFT adapter: %v", err)
	}
}
