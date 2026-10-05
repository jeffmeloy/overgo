package server

import (
	"net/http/httptest"
	"os"
	"strconv"
	"testing"

	"overgo/internal/webuilane"
)

// TestWebUIBrowserPhaseProfile drives three turns: each turn's facts state
// its tokens per forward, the inspector shows the recent turns' decode rate
// and tokens per forward as a trend with the latest values, and the Runtime
// tab lists tokens per forward per slot with the same trend.
func TestWebUIBrowserPhaseProfile(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the phase profile leg uses Chromium through cmd/webui-lane")
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Profiled ", "answer"}}))
	server := httptest.NewServer(handler)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer textarea')`)
	for turn := 1; turn <= 3; turn++ {
		check(`(() => {
  const input = document.querySelector('.composer textarea');
  input.value = 'profile this turn'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
		settle(`document.querySelectorAll('.chat-log .msg.assistant').length === ` + strconv.Itoa(turn) + ` && !document.querySelector('.send-button').disabled`)
	}

	// The turn's facts state its tokens per forward: one, without speculation.
	check(`[...document.querySelectorAll('#conversation-settings .stat')].some(stat => stat.querySelector('.k').textContent === 'Per forward' && stat.querySelector('.v').textContent.startsWith('1.00'))`)

	// The inspector shows the recent turns' trend.
	check(`(() => { [...document.querySelectorAll('.chat-log .msg.assistant button')].filter(button => button.textContent === 'inspect').at(-1).click(); return true; })()`)
	settle(`!!document.querySelector('#inspector .turn-trend')`)
	check(`(() => {
  const lines = [...document.querySelectorAll('#inspector .turn-trend .turn-trend-line')];
  return lines.length === 2 && lines[0].querySelector('.note').textContent.startsWith('Decode: ') && lines[0].querySelector('.note').textContent.endsWith('latest over 3 turns')
    && lines[1].querySelector('.note').textContent === 'Tokens per forward: 1.00 latest over 3 turns' && lines.every(line => line.querySelectorAll('svg circle').length === 3);
})()`)
	check(`[...document.querySelectorAll('#inspector .stat')].some(stat => stat.querySelector('.k').textContent === 'Per forward')`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "turn-trend")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// The Runtime tab lists tokens per forward per slot, with the same trend.
	check(`(() => { location.hash = '#runtime'; return true; })()`)
	settle(`!!document.querySelector('#panels .turn-trend') && [...document.querySelectorAll('#panels table.grid th')].some(cell => cell.textContent.toLowerCase() === 'per forward')`)
	check(`document.querySelector('#panels .turn-trend .turn-trend-line:last-child .note').textContent === 'Tokens per forward: 1.00 latest over 3 turns'`)
	webuilane.Leg(t, "phase profile leg", "each turn's facts stated its tokens per forward, the inspector showed the recent turns' decode rate and tokens per forward as a trend with the latest values, and the Runtime tab listed tokens per forward per slot with the same trend")
}
