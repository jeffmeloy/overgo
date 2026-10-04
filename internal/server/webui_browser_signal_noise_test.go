package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserSignalNoise drives what once said nothing or said it twice:
// the Datasets header names the registry by its short identity with the whole
// of it in a tooltip, never as sixty-four digits, and a dataset's status is a
// tag in the tone every other status uses; an artifact with no payload leads
// with its identity rather than with a note that it has none, and names
// producers only when it has them; each connection light is named beside it.
func TestWebUIBrowserSignalNoise(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the signal and noise leg uses Chromium through cmd/webui-lane")
	}
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	const registry = "dataset:sha256:62551eb59d91e08a531405a4d22be27cd55dd7b89fd0c7d1a95fc4b9901f76f4"
	const manifest = "model:sha256:000df141c3aa0a2c4d3b5d6a5f8cf3a4a61c84d4d68b0f5e8c1c3c1f2a7b9d00"
	const picture = "output:sha256:0477fc0209f1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091"
	const producer = "run:sha256:9a8b7c6d5e4f30211203f4e5d6c7b8a99a8b7c6d5e4f30211203f4e5d6c7b8a9"
	catalogs := map[string]string{
		"/datasets": `{"count":2,"catalog":"` + registry + `","datasets":[
			{"name":"alpha","source":"git","modality":"text","formats":["jsonl"],"files":3,"available":true},
			{"name":"beta","source":"loose","modality":"text","formats":["jsonl"],"files":1,"available":false}]}`,
		"/artifacts": `{"count":2,"truncated":false,"artifacts":[
			{"descriptor":{"id":"` + manifest + `","media_type":"application/vnd.overgo.artifact-manifest+json","size":169},"payload":false,"producers":[]},
			{"descriptor":{"id":"` + picture + `","media_type":"image/png","size":2048},"payload":true,"producers":["` + producer + `"]}]}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, found := catalogs[r.URL.Path]; found && r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
			return
		}
		handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL+"/#datasets")
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

	// Datasets: the registry by its short identity, each status a tag.
	settle(`!!document.querySelector('#panel-datasets.active table.grid td')`)
	check(`(() => {
  const panel = document.querySelector('#panel-datasets'), registry = panel.querySelector('.lens-summary .mono');
  const status = [...panel.querySelectorAll('table.grid tr')].filter(row => row.querySelector('td')).map(row => row.lastElementChild.querySelector('.tag'));
  return registry.textContent === 'dataset:62551eb59d' && registry.title === '` + registry + `' && !/[0-9a-f]{64}/.test(panel.textContent)
    && status[0].textContent === 'available' && status[0].classList.contains('ok') && status[1].textContent === 'missing' && status[1].classList.contains('caution');
})()`)

	// Artifacts: no lead note for a missing payload, no producer line without producers.
	check(`(() => { location.hash = 'artifacts'; return true; })()`)
	settle(`document.querySelectorAll('#panel-artifacts.active article.artifact').length === 2`)
	check(`(() => {
  const [bare, shown] = document.querySelectorAll('#panel-artifacts article.artifact');
  const identity = bare.querySelector('.mono');
  return bare.classList.contains('bare') && !bare.textContent.includes('unavailable') && identity.textContent === 'model:000df141c3'
    && identity.getBoundingClientRect().left - bare.getBoundingClientRect().left < 8 && bare.querySelectorAll('.note').length === 1
    && !shown.classList.contains('bare') && !!shown.querySelector('img') && shown.querySelectorAll('.note').length === 2 && shown.textContent.includes('run:9a8b7c6d5e');
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "signal-noise")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// Settings: each connection light is named beside it.
	check(`(() => { location.hash = 'chat'; return true; })()`)
	settle(`!!document.querySelector('#panel-chat.active .composer textarea')`)
	check(`(() => { [...document.querySelectorAll('button')].find(button => button.textContent === 'Settings').click(); return true; })()`)
	settle(`!!document.querySelector('#settings-dialog[open] .dot-label')`)
	check(`[...document.querySelectorAll('#settings-dialog .dot-label')].map(label => label.textContent.trim()).join('|') === 'server|proxy|device' && !!document.querySelector('#settings-dialog .dot-label #proxy-dot')`)
	webuilane.Leg(t, "signal and noise leg", "the Datasets header named its registry by short identity with the whole in a tooltip and each status was a tag; an artifact with no payload led with its identity and named producers only when it had them; each connection light was named beside it")
}
