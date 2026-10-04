package server

import (
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserSecondaryButtons walks the tabs richest in secondary
// actions and holds every visible secondary button to one outlined style: a
// line around it that differs from the surface, and the accent fill left to
// the primary action; an icon button keeps its line transparent.
func TestWebUIBrowserSecondaryButtons(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the secondary buttons leg uses Chromium through cmd/webui-lane")
	}
	server := httptest.NewServer(newTestHandlerWithRepository(t, &fakeGenerator{}))
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
	// outlined: every visible secondary button has a line unlike its surface; an icon button has none.
	const outlined = `(() => {
  const visible = [...document.querySelectorAll('.btn.alt')].filter(button => button.getClientRects().length && getComputedStyle(button).visibility !== 'hidden');
  const surface = getComputedStyle(document.body).backgroundColor;
  return visible.length > 0 && visible.every(button => {
    const style = getComputedStyle(button);
    if (button.classList.contains('icon-button')) return style.borderTopColor === 'rgba(0, 0, 0, 0)';
    return parseFloat(style.borderTopWidth) >= 1 && style.borderTopColor !== surface && style.borderTopColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor === 'rgba(0, 0, 0, 0)';
  });
})()`
	settle(`!!document.querySelector('.composer textarea')`)
	visited := []string{}
	for _, tab := range []string{"chat", "library", "agent", "runtime"} {
		check(`(() => { location.hash = '` + tab + `'; return true; })()`)
		settle(`!!document.querySelector('#panel-` + tab + `.active') && document.querySelector('#panel-` + tab + `').children.length > 0 && !document.querySelector('#panel-` + tab + ` .loading')`)
		check(outlined)
		visited = append(visited, tab)
	}
	webuilane.Leg(t, "secondary buttons leg", "every visible secondary button on %s had a line unlike its surface and no accent fill, and icon buttons kept their line transparent", strings.Join(visited, ", "))
}
