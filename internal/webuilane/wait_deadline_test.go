package webuilane

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/testskip"
)

// TestWebUIBrowserWaitDeadline holds a leg's waits to the budget the lane
// declares, in a real browser: the browser a leg opens carries the lane's
// budget, and on a page that is alive a predicate that cannot hold ends the
// wait when its budget has passed, with the leg's own context still alive
// and with what the page showed in the failure. The wait that cannot hold is
// given a budget of the test's own, so the leg does not sit out the lane's.
func TestWebUIBrowserWaitDeadline(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "waits on a real page run through cmd/webui-lane")
	}
	path, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	const shown = "a page that is alive and shows this"
	browser, err := Open(ctx, path, `data:text/html,<body><div id="panels">`+shown+`</div></body>`)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	declared := browser.socket.budget
	if declared <= 0 {
		t.Fatalf("the browser a leg opened waits %s for its page: the lane declared it no budget", declared)
	}
	// A wait that holds reports what the page shows as it begins, so the next wait's failure has it.
	if err := browser.Eventually(ctx, "true"); err != nil {
		t.Fatal(err)
	}
	const brief = time.Millisecond
	browser.socket.budget = brief
	err = browser.Eventually(ctx, "false")
	if !errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil || !strings.Contains(err.Error(), brief.String()) || !strings.Contains(err.Error(), shown) {
		t.Fatalf("a predicate that cannot hold ended with %v, the leg's context ended: %t; want the budget's deadline, naming the %s it was given and what the page showed", err, ctx.Err() != nil, brief)
	}
	Leg(t, "wait deadline leg", "a browser opened by a leg carried the lane's budget of %s for each answer from its page, and a predicate that cannot hold on a live page ended at its budget with what the page showed", declared)
}
