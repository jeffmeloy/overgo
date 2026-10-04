package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserSupportTier drives the support tier over a catalog the
// leg's server answers in the catalog's shape: the served model's tier
// stands beside the header pill with every rung in its title, the model
// details list the rungs, each picker row carries its tier, and a turn's
// inspector grades the model that ran it.
func TestWebUIBrowserSupportTier(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the support tier leg uses Chromium through cmd/webui-lane")
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Tier answer"}}))
	var health struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/health", "").Body.Bytes(), &health); err != nil || health.Model == "" {
		t.Fatalf("health = %+v, %v", health, err)
	}
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.Bytes(), &manifest); err != nil || manifest.Model == nil || manifest.Model.Model == "" {
		t.Fatalf("manifest = %+v, %v", manifest, err)
	}
	rung := func(name string, held bool, detail string) map[string]any {
		return map[string]any{"name": name, "held": held, "detail": detail}
	}
	catalog, err := json.Marshal(map[string]any{"truncated": false, "models": []map[string]any{
		{
			"model": manifest.Model.Model, "location": "C:/models/" + health.Model, "present": true, "recipe": "recipe:served",
			"capabilities": []map[string]any{{"task": "inference", "recipe": "recipe:served", "tier": "verified"}},
			"support": map[string]any{"tier": "self-consistent", "rungs": []map[string]any{
				rung("registered", true, "the store registers its weights at C:/models/"+health.Model),
				rung("self-consistent", true, "its verification suite replays its own greedy output: deterministic, not proven correct"),
				rung("parity-checked", false, "no evidence matching a reference implementation"),
				rung("guarded", true, "long-form guard passed at commit 0123456789ab on inference surface feedfacecafe; a later code change can expire it"),
				rung("trainable", false, "no training activation"),
			}},
		},
		{
			"model": "model:bare", "location": "C:/models/bare.gguf", "present": true, "recipe": "recipe:bare",
			"capabilities": []map[string]any{{"task": "inference", "recipe": "recipe:bare", "tier": "verified"}},
			"support": map[string]any{"tier": "registered", "rungs": []map[string]any{
				rung("registered", true, "the store registers its weights at C:/models/bare.gguf"),
				rung("self-consistent", false, "no verified inference activation"),
				rung("parity-checked", false, "no evidence matching a reference implementation"),
				rung("guarded", false, "no long-form record"),
				rung("trainable", false, "no training activation"),
			}},
		},
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
	settle(`!!document.querySelector('.composer textarea')`)

	// The served model's tier stands beside its pill, every rung in the title; the details list them.
	settle(`(() => { const tier = document.getElementById('model-tier'); return !tier.hidden && tier.textContent === 'self-consistent · guarded' && tier.title.includes('✓ guarded: long-form guard passed') && tier.title.includes('✗ parity-checked'); })()`)
	check(`(() => { const rungs = [...document.querySelectorAll('#model-support li')]; return rungs.length === 5 && rungs.filter(rung => rung.classList.contains('held')).length === 3 && rungs[1].textContent.includes('not proven correct'); })()`)

	// Each picker row carries its tier.
	check(`(() => { document.getElementById('model-pill').click(); return true; })()`)
	settle(`[...document.querySelectorAll('dialog[open] .row[tabindex]')].length === 2`)
	check(`(() => {
  const tier = (name) => ([...document.querySelectorAll('dialog[open] .row[tabindex]')].find(row => row.querySelector('.mono').textContent === name) || document.body).querySelector('.support-tier');
  return tier(` + "`" + health.Model + "`" + `).textContent === 'self-consistent · guarded' && tier('bare.gguf').textContent === 'registered' && tier('bare.gguf').classList.contains('caution');
})()`)
	// The served row's marker is the one control tag of the card, and a parity-checked tier takes the name's column, not the control's.
	check(`(() => {
  const rows = [...document.querySelectorAll('dialog[open] .row[tabindex]')];
  const probe = document.createElement('span'); probe.className = 'tag support-tier ok'; rows[0].appendChild(probe);
  const column = getComputedStyle(probe).gridColumnStart; probe.remove();
  return rows.filter(row => row.querySelector('.tag.ok:not(.support-tier)')).length === 1 && column === '1';
})()`)
	check(`(() => { document.querySelector('dialog[open]').dispatchEvent(new Event('cancel')); return true; })()`)
	settle(`!document.querySelector('dialog[open]')`)

	// A turn's inspector grades the model that ran it.
	check(`(() => {
  const input = document.querySelector('.composer textarea');
  input.value = 'grade the model'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.chat-log .msg.assistant')].some(message => message.textContent.includes('Tier answer') && [...message.querySelectorAll('button')].some(button => button.textContent === 'inspect'))`)
	check(`(() => { [...document.querySelectorAll('.chat-log .msg.assistant button')].find(button => button.textContent === 'inspect').click(); return true; })()`)
	settle(`(() => { const support = document.querySelector('#inspector .turn-support'); return !!support && support.textContent.startsWith('Model support now') && support.querySelectorAll('li').length === 5 && support.querySelectorAll('li.held').length === 3; })()`)
	webuilane.Leg(t, "support tier leg", "the served model's tier stood beside its pill with every rung in its title, the model details listed the rungs, each picker row carried its tier, and a turn's inspector graded the model that ran it")
}
