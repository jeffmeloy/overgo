package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserMemoryEstimate drives the measured memory before launch:
// a picker row states its model's measured peak against the device it ran
// on, after-load bytes and residency, with the peak at each prompt length its
// long-form run climbed in the details; a row no run measured states none;
// and the training tab says its runs execute on the host.
func TestWebUIBrowserMemoryEstimate(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the memory estimate leg uses Chromium through cmd/webui-lane")
	fixture := newDPOTrainingFixture(t)
	handler, err := New(Config{Repository: fixture.store, Workflows: fixture.workspace}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	const gibibyte = 1 << 30
	catalog, err := json.Marshal(map[string]any{"truncated": false, "models": []map[string]any{
		{
			"model": "model:measured", "location": "C:/models/measured.gguf", "present": true, "recipe": "recipe:measured",
			"memory": map[string]any{
				"load_bytes": 5 * gibibyte, "peak_bytes": 6 * gibibyte, "device_bytes": 24 * gibibyte, "residency": "device-native",
				"context":              []map[string]any{{"prompt_tokens": 1024, "peak_bytes": 6.5 * gibibyte}, {"prompt_tokens": 4096, "peak_bytes": 7 * gibibyte}},
				"context_device_bytes": 24 * gibibyte,
			},
		},
		{"model": "model:unmeasured", "location": "C:/models/unmeasured.gguf", "present": true, "recipe": "recipe:unmeasured"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/catalog/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(catalog)
			return
		}
		handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.getElementById('model-pill') && !document.getElementById('model-pill').hidden`)

	// A picker row states its measured memory before serving; the curve is in its details.
	check(`(() => { document.getElementById('model-pill').click(); return true; })()`)
	settle(`[...document.querySelectorAll('dialog[open] .row[tabindex]')].length === 2`)
	check(`(() => {
  const row = (name) => [...document.querySelectorAll('dialog[open] .row[tabindex]')].find(item => item.querySelector('.mono').textContent === name);
  const lines = (name) => [...row(name).querySelectorAll('.model-memory')].map(line => line.textContent);
  const measured = lines('measured.gguf');
  return measured.length === 2 && measured[0] === 'measured peak 6.00 GB of 24.00 GB · 5.00 GB after load · device-native' &&
    row('measured.gguf').querySelector(':scope > .model-memory').textContent === measured[0] &&
    row('measured.gguf').querySelector('details .model-memory').textContent === 'peak by prompt length: 1,024 tokens 6.50 GB · 4,096 tokens 7.00 GB of 24.00 GB' &&
    lines('unmeasured.gguf').length === 0;
})()`)
	check(`(() => { document.querySelector('dialog[open]').dispatchEvent(new Event('cancel')); return true; })()`)
	settle(`!document.querySelector('dialog[open]')`)

	// The training tab says its runs execute on the host.
	check(`(() => { location.hash = 'training-jobs'; return true; })()`)
	settle(`(() => { const note = document.querySelector('#panel-training-jobs.active .workflow-placement'); return !!note && !note.hidden && note.textContent === 'Runs on the host processor: it needs no device memory.'; })()`)
	webuilane.Leg(t, "memory estimate leg", "a picker row stated its model's measured peak against the device it ran on with after-load bytes and residency and its peak by prompt length in the details, an unmeasured row stated none, and the training tab said its runs execute on the host")
}
