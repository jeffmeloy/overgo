package server

import (
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserSecondaryButtons walks the tabs richest in secondary
// actions and holds every visible secondary button to one outlined style: a
// line around it that differs from the surface, and the accent fill left to
// the primary action; an icon button keeps its line transparent.
func TestWebUIBrowserSecondaryButtons(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the secondary buttons leg uses Chromium through cmd/webui-lane")
	server := httptest.NewServer(newTestHandlerWithRepository(t, &fakeGenerator{}))
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
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
