package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/testskip"
	"overgo/internal/tokenizer"
	"overgo/internal/webuilane"
)

// analysisViewsGenerator declares hidden-state and attention capture, so the
// workbench offers the analysis views; the leg's server answers their
// captures, which need a real model, with fixed results in the handlers'
// shape.
type analysisViewsGenerator struct{ *fakeGenerator }

func (analysisViewsGenerator) SupportsHiddenStateCapture() bool { return true }
func (analysisViewsGenerator) AttentionCaptureLayers() []int32  { return []int32{0} }
func (analysisViewsGenerator) ExtractAttention(context.Context, []tokenizer.TokenID, int32) (inference.AttentionCapture, error) {
	return inference.AttentionCapture{}, errors.New("the analysis views leg answers attention itself")
}

const (
	analysisStatesResult    = `{"tokens":[{"id":1,"text":"quick"},{"id":2,"text":"brown"},{"id":3,"text":"fox"}],"positions":3,"requested_positions":3,"max_positions":48,"truncated":false,"layer":0,"block_count":1,"width":4,"metric":"spearman","k":1,"stress":0.1,"layout_iterations":3,"distance":[[0,0.4,0.9],[0.4,0,0.5],[0.9,0.5,0]],"neighbors":[[1],[0],[1]],"layout":[[0,0],[1,0],[2,1]]}`
	analysisAttentionResult = `{"tokens":[{"id":1,"text":"quick"},{"id":2,"text":"brown"},{"id":3,"text":"fox"}],"positions":3,"requested_positions":3,"max_positions":32,"truncated":false,"layer":0,"block_count":1,"head_dim":2,"heads":1,"kv_heads":1,"group_size":1,"scale":0.7071,"weights":[[[1,0,0],[0.5,0.5,0],[0.2,0.3,0.5]]]}`
)

// TestWebUIBrowserAnalysisViews drives the analysis views: a capture the
// operator cancels rejects its request as aborted, a capture that answers
// draws its matrix as a heatmap with a colour-scale legend and the prompt's
// tokens on its axes (hidden states and attention alike), a signed series
// keeps negative measurements below a visible zero axis, the model facts
// are fetched once however many views read them, and the status is probed
// again when the page becomes visible.
func TestWebUIBrowserAnalysisViews(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the analysis views leg uses Chromium through cmd/webui-lane")
	}
	handler := newTestHandler(t, analysisViewsGenerator{&fakeGenerator{}})
	var modelReads, healthReads, statesCaptures atomic.Int64
	// held: the first capture's request, released when the leg ends.
	held := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/analyze/model":
			modelReads.Add(1)
		case "/health":
			healthReads.Add(1)
		case "/analyze/states":
			// The first capture is held until the page abandons it.
			if statesCaptures.Add(1) == 1 {
				select {
				case <-r.Context().Done():
				case <-held:
				}
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(analysisStatesResult))
			return
		case "/analyze/attention":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(analysisAttentionResult))
			return
		}
		handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	// Released before the server closes, which waits for every request.
	defer close(held)
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
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => {
  window.analysisButton = (label) => [...document.querySelectorAll('#panel-inspect button')].find(button => button.textContent === label && !button.hidden);
  window.analysisHeatmap = () => {
    const legend = [...document.querySelectorAll('#panel-inspect svg')].find(svg => svg.querySelector('linearGradient stop'));
    const ticks = [...document.querySelectorAll('#panel-inspect svg text')].map(text => text.textContent);
    return !!legend && legend.querySelectorAll('linearGradient stop').length > 1 && ['quick', 'brown', 'fox'].every(token => ticks.includes(token));
  };
  return true;
})()`)

	// A cancelled capture aborts its request.
	check(`(() => { location.hash = 'states'; return true; })()`)
	settle(`!!analysisButton('capture')`)
	check(`(() => { analysisButton('capture').click(); return true; })()`)
	settle(`!!analysisButton('cancel')`)
	check(`(() => { analysisButton('cancel').click(); return true; })()`)
	// The surface shows [cancelled] only when the held request rejects as
	// aborted, which the view's forwarded signal alone causes.
	settle(`document.querySelector('#panel-inspect').textContent.includes('[cancelled]') && !analysisButton('cancel')`)
	if captures := statesCaptures.Load(); captures != 1 {
		t.Fatalf("the cancelled capture made %d requests", captures)
	}

	// A capture that answers draws its heatmap with a legend and token axes.
	check(`(() => { analysisButton('capture').click(); return true; })()`)
	settle(`analysisHeatmap()`)
	check(`(() => { location.hash = 'attention'; return true; })()`)
	settle(`!!analysisButton('capture') && !document.querySelector('#panel-inspect').textContent.includes('Distance matrix')`)
	check(`(() => { analysisButton('capture').click(); return true; })()`)
	settle(`analysisHeatmap() && document.querySelector('#panel-inspect').textContent.includes('Attention weights')`)

	// A signed series keeps its negative measurements below the zero axis.
	check(`(() => {
  const chart = overgo.viz.signedSeries([{ label: 'margin', values: [-2, 1, -0.5] }]);
  const axis = Number(chart.querySelector('.zero-axis').getAttribute('y1'));
  const dots = [...chart.querySelectorAll('circle')];
  return dots.map(dot => dot.getAttribute('data-value')).join(',') === '-2,1,-0.5' &&
    dots.filter(dot => Number(dot.getAttribute('data-value')) < 0).every(dot => Number(dot.getAttribute('cy')) > axis);
})()`)

	// The model facts are read once for every view that uses them.
	check(`(() => { location.hash = 'model'; return true; })()`)
	settle(`document.querySelector('#panel-model.active, #panel-inspect.active') !== null`)
	check(`(async () => { await overgo.modelInfo(); await overgo.modelInfo(); return true; })()`)
	if reads := modelReads.Load(); reads != 1 {
		t.Fatalf("the model facts were read %d times", reads)
	}

	// A page that becomes visible probes the status again.
	before := healthReads.Load()
	check(`(() => { document.dispatchEvent(new Event('visibilitychange')); return true; })()`)
	settle(`!!document.querySelector('.composer textarea')`)
	for healthReads.Load() == before {
		if err := ctx.Err(); err != nil {
			t.Fatal("a visible page did not probe the status again")
		}
		check(`true`)
	}
	webuilane.Leg(t, "analysis views leg", "a cancelled capture rejected its request as aborted, heatmaps drew a legend and token axes for hidden states and attention, a signed series kept negatives below its zero axis, the model facts were read once and a visible page probed the status")
}
