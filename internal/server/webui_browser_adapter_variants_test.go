package server

import (
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/peft"
	"overgo/internal/trainingworkflow"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserAdapterVariants holds the two adapter variants the Train
// tab offers for a LoRA run. Fine-tuning the whole model shows neither. The
// adapter shows rslora, a choice left unset, and lora plus, a rate left
// empty, and the note beside them says what each is. A run with rslora
// chosen and no rate typed finishes with its held-out verdict, and the
// adapter it wrote is scaled as rslora scales one and by no rate, which is
// what shows the page sent the one and not the other. The form with the two
// controls has no layout finding at desktop and phone widths in either
// colour scheme.
func TestWebUIBrowserAdapterVariants(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the adapter variants leg uses Chromium through cmd/webui-lane")
	fixture := newSFTFixture(t)
	handler := newTestHandlerForRepository(t, fixture.store, &fakeGenerator{}, func(config *Config) { config.Workflows = fixture.workspace })
	server := httptest.NewServer(handler)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	const helpers = `(() => {
  window.trainingPanel = () => document.querySelector('#panel-training-jobs');
  window.control = (label) => [...trainingPanel().querySelectorAll('label.control')].find((item) => item.querySelector('span').textContent === label);
  window.press = (text) => [...trainingPanel().querySelectorAll('button')].find((button) => button.textContent === text).click();
  window.verdict = () => trainingPanel().querySelector('[aria-label="Held-out verdict"]');
  window.failed = () => /^failed/.test(trainingPanel().querySelector('[aria-label="run status"]').textContent);
  return true;
})()`
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
	check(helpers)

	// The whole model has no adapter to vary.
	check(`!control('rslora') && !control('lora plus')`)
	// The adapter brings both, unset, and says what they are.
	check(`(() => {
  const select = trainingPanel().querySelector('select[aria-label="capability"]');
  select.value = [...select.options].find((option) => /LoRA/.test(option.text)).value;
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	settle(`!!control('rslora') && !!control('lora plus')`)
	check(`(() => {
  const choice = control('rslora').querySelector('select'), rate = control('lora plus').querySelector('input');
  const note = trainingPanel().querySelector('.workflow-placement').textContent;
  if (!choice || choice.value !== '' || ![...choice.options].some((option) => option.value === 'true')) throw new Error('rslora is not a choice left unset that offers true');
  if (!rate || rate.type !== 'number' || rate.value !== '') throw new Error('lora plus is not a number left empty');
  if (!/rslora/.test(note) || !/lora plus/.test(note)) throw new Error('the note beside the controls names neither variant: ' + note);
  return true;
})()`)
	for _, scheme := range webuilane.ColourSchemes {
		if err := browser.SetColorScheme(ctx, scheme); err != nil {
			t.Fatal(err)
		}
		for _, viewport := range webuilane.ScreenViewports {
			findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "adapter-variants-"+scheme)
			if err != nil || len(findings) != 0 {
				t.Fatalf("the Train tab with the adapter's variants, at %s in the %s scheme: %v %v", viewport.Name, scheme, findings, err)
			}
		}
	}

	// A run with rslora chosen and no rate typed.
	settle(fmt.Sprintf(`[...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q)`, fixture.conversations.String()))
	check(fmt.Sprintf(`(() => {
  const select = control('conversations').querySelector('select');
  select.value = %q;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  control('steps').querySelector('input').value = '%d';
  control('rank').querySelector('input').value = '2';
  control('output').querySelector('input').value = 'variant';
  const choice = control('rslora').querySelector('select');
  choice.value = 'true';
  choice.dispatchEvent(new Event('input', { bubbles: true }));
  press('Run');
  return select.value === %q;
})()`, fixture.conversations.String(), adapterSteps, fixture.conversations.String()))
	settle(`!!verdict() || failed()`)
	check(`(() => { if (!verdict()) throw new Error('the run ended without its verdict: ' + trainingPanel().querySelector('[aria-label="run status"]').textContent); return true; })()`)
	adapter, err := peft.Load(filepath.Join(fixture.roots.Checkpoints, "variant", trainingworkflow.LoRAAdapterDirectory))
	if err != nil {
		t.Fatal(err)
	}
	// Rank 2 at the workbench's alpha a rank, scaled by no lora plus rate.
	if want := float64(2 * trainingworkflow.LoRAAlphaPerRank); !adapter.Config.RSLoRA || adapter.Config.Alpha != want {
		t.Fatalf("the adapter the run wrote has rslora %t and alpha %v, want rslora and alpha %v: the page sent rslora and no rate", adapter.Config.RSLoRA, adapter.Config.Alpha, want)
	}
	webuilane.Leg(t, "adapter variants leg", "the Train tab offered rslora and a lora plus rate for a LoRA adapter and for no whole-model run, and a run with rslora chosen wrote an adapter scaled as rslora scales one")
}
