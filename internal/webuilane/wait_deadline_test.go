package webuilane

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestWebUIBrowserWaitDeadline holds a leg's waits, in a real browser, to the owner's rule that no clock of the lane's
// ends them (no-clocks/own-budgets): the browser a leg opens carries no budget unless the user set one with
// webui-lane -wait, and the page's answer ends a wait; and on a page that is alive a predicate that cannot hold ends,
// under a limit the user set, when that limit has passed, with the leg's own context still alive and with what the
// page showed in the failure. The limit is the test's own, so the leg does not wait on a page that never answers.
func TestWebUIBrowserWaitDeadline(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "waits on a real page run through cmd/webui-lane")
	}
	path, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	testcheck.NoError(t, err)
	ctx := t.Context()
	const shown = "a page that is alive and shows this"
	browser := testcheck.Must(Open(ctx, path, `data:text/html,<body><div id="panels">`+shown+`</div></body>`))(t)
	defer browser.Close()
	if set := os.Getenv(WaitBudgetEnvironment); set == "" && browser.socket.budget != 0 {
		t.Fatalf("the browser a leg opened waits %s for its page, though no one set a limit", browser.socket.budget)
	}
	// A wait the page answers ends at the answer, and reports what the page shows as it begins, so the next wait's
	// failure has it.
	testcheck.NoError(t, browser.Eventually(ctx, "true"))
	const brief = time.Millisecond
	browser.socket.budget = brief
	err = browser.Eventually(ctx, "false")
	if !errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil || !strings.Contains(err.Error(), brief.String()) || !strings.Contains(err.Error(), shown) {
		t.Fatalf("a predicate that cannot hold ended with %v, the leg's context ended: %t; want the limit's deadline, naming the %s set and what the page showed", err, ctx.Err() != nil, brief)
	}
	Leg(t, "wait deadline leg", "a browser opened by a leg carried no limit no one set, the page's answer ended its wait, and under a limit set of %s a predicate that cannot hold on a live page ended at it with what the page showed", brief)
}
