package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/modeldevice"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserMonitor drives the monitor on the Runtime tab, which reads
// /metrics once a second while the tab shows: it draws the served card's
// load, memory, temperature, power and link, where the experts are and the
// share of routed reads that found theirs on the card; a change on the card
// draws on a later read, and a hit share or a prefill speed is the change
// between two reads; a card with no load readings and a model with no
// file-backed experts draw none of theirs. The Requests view shows the
// generation's reused prompt tokens and its prefill and decode speed.
func TestWebUIBrowserMonitor(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the monitor leg uses Chromium through cmd/webui-lane")
	// The Runtime tab serves over a store.
	generator := testMonitoredGenerator()
	handler := newTestHandlerWithRepository(t, generator)
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	plainHandler := newTestHandlerWithRepository(t, &memoryOwnersGenerator{fakeGenerator: &fakeGenerator{}, memory: testDeviceMemory()})
	defer plainHandler.Close()
	plain := httptest.NewServer(plainHandler)
	defer plain.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#runtime")
	defer browser.Close()
	check(`(() => {
  window.monitorShows = () => Object.fromEntries([...document.querySelectorAll('#panels .monitor .stat')].map((stat) => [stat.querySelector('.k').textContent, stat.querySelector('.v').textContent]));
  return true;
})()`)

	// The card's load and memory and the experts' places draw from the first read; the hit share is since the model loaded,
	// and a second read starts the trends of the decode speed, the card's load and memory and the hit share.
	settle(`(() => {
  const shown = monitorShows();
  return shown['GPU load'] === '87 %' && shown['Card memory'] === '14.00 GB of 24.00 GB' && shown['Temperature'] === '64 °C'
    && shown['Power'] === '313 W of 425 W' && shown['PCIe'] === 'Gen4 x16 2.33 GB/s to the card'
    && shown['Experts on the card'] === '1,536 in 6.00 GB' && shown['Experts in RAM'] === '6,144 in 24.00 GB'
    && shown['Expert hits'] === '75 % on the card' && shown['Prefill'] === '—'
    && document.querySelectorAll('#panels .monitor .turn-trend-line').length === 4;
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "monitor")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// A change on the card draws on a later read: the hit share is the routed reads between two reads, 900 of 1,000.
	memory := testDeviceMemory()
	memory.FreeBytes = 4 * gibibyte
	memory.Experts = &modeldevice.ExpertPlacement{
		CardExperts: 1536, HostExperts: 6144, CardBytes: 6 * gibibyte, HostBytes: 24 * gibibyte, Routes: 5000, CardRoutes: 3900,
	}
	generator.set(memory)
	generator.setCard([]driver.CardReading{{Name: "utilization_percent", Value: 42}})
	settle(`(() => {
  const shown = monitorShows();
  return shown['GPU load'] === '42 %' && shown['Card memory'] === '20.00 GB of 24.00 GB' && shown['Expert hits'] === '90 % on the card'
    && !('Temperature' in shown) && !('PCIe' in shown);
})()`)

	// A prompt of three tokens, one reused, prefilled in 2 ms: two tokens over 2 ms between two reads.
	check(`(async () => {
  const answer = await fetch('/v1/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: 'test-model', prompt: 'hi', max_tokens: 2, temperature: 0 }) });
  return answer.status === 200;
})()`)
	settle(`monitorShows()['Prefill'] === '1000.0 tok/s'`)

	// The Requests view shows the generation's reused tokens and its speeds.
	check(`(() => { location.hash = 'requests'; return true; })()`)
	settle(`[...document.querySelectorAll('#panel-api tr')].some((row) => row.textContent.includes('/v1/completions') && row.textContent.includes('3 in (1 reused, 33%) · 2 out') && row.textContent.includes('prefill 1000.0 tok/s · decode '))`)

	// A card with no load readings and a model with no file-backed experts draw the card's memory alone.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+plain.URL+"/#runtime`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + plain.URL + "`" + ` && !!document.querySelector('#panels .monitor .stat')`)
	check(`(() => {
  const labels = [...document.querySelectorAll('#panels .monitor .stat .k')].map((label) => label.textContent).join();
  return labels === 'Decode,Prefill,Card memory' && window.overgo.errors.length === 0;
})()`)
	webuilane.Leg(t, "monitor leg", "the Runtime tab drew the card's load, memory, temperature, power and link, the experts on the card and in RAM and the share of routed reads found on the card; a change on the card drew on a later read, with the hit share and the prefill speed taken between two reads; the Requests view showed the reused tokens and the prefill and decode speed; a card with no readings and a model with no file-backed experts drew the card's memory alone")
}
