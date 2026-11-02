package server

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"overgo/internal/overgodb"
)

// TestWebUIBrowserLongSessionStaysBounded holds what a long session costs the browser: a conversation of many messages
// renders each once, and a tool's streamed output costs in proportion to what
// arrives, keeping a bounded tail on the page: 4 MiB of it took 32.9 seconds before each piece became its own text node.
func TestWebUIBrowserLongSessionStaysBounded(t *testing.T) {
	t.Parallel()
	const (
		assistants         = 500
		keptOutput         = 1<<20 + 4096
		outputMilliseconds = 5000
	)
	requireWebUILane(t, "the long session leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	server := httptest.NewServer(newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{})))
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!window.overgo && !!window.overgo.thread`)
	check(`(() => {
  (async () => {
    const heap = () => performance.memory ? performance.memory.usedJSHeapSize : 0;
    const host = document.createElement('div');
    host.style.height = '400px';
    document.body.append(host);
    const thread = overgo.thread(host);
    const realMD = overgo.md;
    let rendered = 0;
    overgo.md = (text) => { rendered++; return realMD(text); };
    const heapBefore = heap();
    const started = performance.now();
    const paragraph = 'a word of an answer '.repeat(30);
    for (let index = 0; index < 1000; index++) {
      if (index % 50 === 0) await new Promise((resume) => setTimeout(resume));
      thread.add(index % 2 ? 'assistant' : 'user', index + ' ' + paragraph, { response: 'response-' + index });
    }
    const conversation = performance.now() - started;
    const nodes = host.querySelectorAll('*').length;
    const card = thread.toolCard({ name: 'tool', arguments: {} });
    const chunk = 'x'.repeat(4096);
    const streaming = performance.now();
    for (let index = 0; index < 4096; index++) {
      if (index % 64 === 0) await new Promise((resume) => setTimeout(resume));
      card.output(chunk);
    }
    const output = performance.now() - streaming;
    const shown = host.querySelector('.tool-output').textContent.length;
    overgo.md = realMD;
    window.measured = { conversation, nodes, rendered, output, shown, heapBefore, heapAfter: heap() };
  })().catch((err) => { window.measured = { failed: String(err) }; });
  return true;
})()`)
	settle(`!!window.measured`)
	var text string
	if err := browser.Evaluate(context.WithoutCancel(ctx), `JSON.stringify(window.measured)`, &text); err != nil {
		t.Fatal(err)
	}
	var measured struct {
		Conversation, Output   float64
		Nodes, Rendered, Shown int
		HeapBefore, HeapAfter  int
		Failed                 string
	}
	if err := json.Unmarshal([]byte(text), &measured); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", text)
	if measured.Failed != "" {
		t.Fatal(measured.Failed)
	}
	if measured.Rendered != assistants {
		t.Fatalf("%d assistant messages rendered %d times", assistants, measured.Rendered)
	}
	if measured.Shown > keptOutput || measured.Output > outputMilliseconds {
		t.Fatalf("16 MiB of tool output kept %d characters and took %.0f ms", measured.Shown, measured.Output)
	}
}
