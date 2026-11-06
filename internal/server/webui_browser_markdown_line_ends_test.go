package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/overgodbtest"
)

// TestWebUIBrowserMarkdownLineEnds holds the message renderer to every line ending a model can write. A heading or a
// fence whose line ended in CR, U+2028 or U+2029 matched no pattern while the paragraph's guard took it for a block, so
// the walk never advanced: 5 million nodes in 1.5 seconds, and a reload repeated it. A long run of unclosed brackets, a
// model stuck repeating, stays quick, even with one close at its very end.
func TestWebUIBrowserMarkdownLineEnds(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the markdown leg uses Chromium through cmd/webui-lane")
	store := overgodbtest.Open(t, t.TempDir())
	server := httptest.NewServer(newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{})))
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!window.overgo && !!window.overgo.md`)
	for name, expression := range map[string]string{
		"a heading ending in CR":          `window.overgo.md("# Title\rnext").querySelector('h1').textContent === 'Title'`,
		"a fence ending in CR":            "window.overgo.md('```go\\rcode\\r```').querySelector('code').textContent === 'code'",
		"a heading ending in U+2028":      `window.overgo.md("# Title next").querySelector('h1').textContent === 'Title'`,
		"a heading ending in U+2029":      `window.overgo.md("# Title next").querySelector('h1').textContent === 'Title'`,
		"CRLF between paragraphs":         `window.overgo.md("one\r\n\r\ntwo").querySelectorAll('p').length === 2`,
		"a list after a CR":               `window.overgo.md("intro\r- item").querySelectorAll('li').length === 1`,
		"a table ending in CR":            `window.overgo.md("| a |\r| - |\r| 1 |").querySelectorAll('td').length === 1`,
		"a lone CR and a lone U+2028":     `window.overgo.md("\r ").childElementCount === 0`,
		"unclosed brackets":               `(() => { const started = performance.now(); window.overgo.md('['.repeat(160000)); return performance.now() - started < 2000; })()`,
		"unclosed brackets, one close":    `(() => { const started = performance.now(); window.overgo.md('['.repeat(160000) + ']'); return performance.now() - started < 2000; })()`,
		"a link still links":              `window.overgo.md('[text](https://example.com)').querySelector('a').textContent === 'text'`,
		"nothing is lost beside a marker": `window.overgo.md("# Title\rnext").textContent.includes('next')`,
	} {
		t.Log(name)
		check(expression)
	}
}
