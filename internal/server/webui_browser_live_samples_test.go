package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"sync"
	"testing"

	"overgo/internal/operation"
	"overgo/internal/webuilane"
)

// heldSFT is the fine-tuning workspace whose run stops each time it has
// reported its samples, until release gives it one more round or is closed:
// held, the run is still going, with the rounds so far reported.
type heldSFT struct {
	*SFTWorkspace
	release <-chan struct{}
}

// ExecuteWorkflow runs the fine-tune under a reporter that holds it after
// each text.
func (workspace heldSFT) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	return workspace.SFTWorkspace.ExecuteWorkflow(ctx, kind, capability, raw, heldReporter{Reporter: reporter, ctx: ctx, release: workspace.release})
}

// heldReporter passes a run's reports on and, having passed on a text, waits
// to be let on, for the release to be closed, or for the run to be ended.
type heldReporter struct {
	operation.Reporter
	ctx     context.Context
	release <-chan struct{}
}

// Text passes the text on, then holds the run.
func (reporter heldReporter) Text(text string) {
	reporter.Reporter.Text(text)
	select {
	case <-reporter.release:
	case <-reporter.ctx.Done():
	}
}

// TestWebUIBrowserLiveSamples holds the Train tab to showing a sampled run's
// samples while the run goes, not when it ends. A conversation fine-tune that
// samples every fourth update is held after the round it takes before
// training. The page then shows that round under each held-out prompt while
// the run's status reads running and no verdict is there; reloaded, it shows
// the same strip from the run's snapshot, with Run disabled; let on by one
// round, it shows the strip grown by that round, in one section and not two;
// and when the run is let go and ends, the verdict stands beside exactly one
// samples section, in step order from before training to the last update.
// The tab with the strip of a running run has no layout finding at desktop
// and phone widths in either colour scheme.
func TestWebUIBrowserLiveSamples(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the live samples leg uses Chromium through cmd/webui-lane")
	fixture := newSFTFixture(t)
	release := make(chan struct{})
	let := sync.OnceFunc(func() { close(release) })
	defer let()
	handler := newTestHandlerForRepository(t, fixture.store, &fakeGenerator{}, func(config *Config) {
		config.Workflows = heldSFT{SFTWorkspace: fixture.workspace, release: release}
	})
	server := httptest.NewServer(handler)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	const helpers = `(() => {
  window.trainingPanel = () => document.querySelector('#panel-training-jobs');
  window.control = (label) => [...trainingPanel().querySelectorAll('label.control')].find((item) => item.querySelector('span').textContent === label);
  window.runButton = () => [...trainingPanel().querySelectorAll('button')].find((button) => button.textContent === 'Run');
  window.runStatus = () => trainingPanel().querySelector('[aria-label="run status"]').textContent;
  window.sampleSections = () => [...trainingPanel().querySelectorAll('[aria-label="Training samples"]')];
  window.sampleSteps = () => sampleSections().flatMap((section) => [...section.querySelectorAll('[data-record]')]).map((strip) => [...strip.querySelectorAll('li')].map((item) => item.dataset.step).join(','));
  window.verdict = () => trainingPanel().querySelector('[aria-label="Held-out verdict"]');
  return true;
})()`
	open := func() {
		t.Helper()
		check(`(() => { location.hash = 'training-jobs'; return true; })()`)
		settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
		check(helpers)
	}
	// running holds the one strip section of a run still going to the rounds it has sampled after.
	running := func(rounds string) string {
		return `(() => {
  const steps = sampleSteps();
  if (sampleSections().length !== 1 || steps.length === 0 || !steps.every((list) => list === '` + rounds + `')) throw new Error('the samples of the running run read ' + JSON.stringify(steps) + ' in ' + sampleSections().length + ' sections, want ` + rounds + ` in one');
  if (![...sampleSections()[0].querySelectorAll('li .preview-text')].every((text) => text.textContent !== '')) throw new Error('a sample of the running run has no text');
  if (!runStatus().startsWith('running') || verdict()) throw new Error('the run is not still going: ' + runStatus());
  return true;
})()`
	}
	open()
	settle(fmt.Sprintf(`!!control('sample every') && [...control('conversations').querySelectorAll('select option')].some((option) => option.value === %q)`, fixture.conversations.String()))
	check(fmt.Sprintf(`(() => {
  const select = control('conversations').querySelector('select');
  select.value = %q;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  control('steps').querySelector('input').value = '12';
  control('sample every').querySelector('input').value = '4';
  control('output').querySelector('input').value = 'live';
  runButton().click();
  return select.value === %q;
})()`, fixture.conversations.String(), fixture.conversations.String()))

	// While the run goes, the round taken before training is on the page.
	settle(`sampleSections().length > 0`)
	check(running("0"))
	for _, scheme := range webuilane.ColourSchemes {
		if err := browser.SetColorScheme(ctx, scheme); err != nil {
			t.Fatal(err)
		}
		for _, viewport := range webuilane.ScreenViewports {
			findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "live-samples-"+scheme)
			if err != nil || len(findings) != 0 {
				t.Fatalf("the Train tab with a running run's samples, at %s in the %s scheme: %v %v", viewport.Name, scheme, findings, err)
			}
		}
	}

	// Reloaded, the page follows the same run and shows the same strip from its snapshot.
	if err := browser.Reload(ctx); err != nil {
		t.Fatal(err)
	}
	open()
	settle(`sampleSections().length > 0`)
	check(running("0"))
	check(`runButton().disabled`)

	// Let on by one round, the strip grows in place: one section, not a second beside the first.
	select {
	case release <- struct{}{}:
	case <-ctx.Done():
		t.Fatalf("the held run did not take another round: %v", ctx.Err())
	}
	settle(`sampleSteps().length > 0 && sampleSteps().every((list) => list !== '0')`)
	check(running("0,4"))

	// Let go, the run ends: the verdict, and one samples section in step order.
	let()
	settle(`!!verdict() && runStatus().startsWith('completed')`)
	settle(`sampleSteps().length > 0 && sampleSteps().every((list) => list === '0,4,8,12')`)
	check(`(() => { if (sampleSections().length !== 1) throw new Error('the finished run shows ' + sampleSections().length + ' samples sections'); return true; })()`)
	webuilane.Leg(t, "live samples leg", "a held fine-tune's samples from before training stood on the Train tab while the run was going and after a reload, and the finished run showed its verdict beside one samples section in step order")
}
