package webuilane

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

// TestDeclaredBudgetEndsAWaitThePageNeverAnswers holds every wait on a page
// to the budget its lane declared. A page that is alive and never answers
// held its test until someone killed the run, and what the page showed was
// never printed, since a killed test prints nothing.
//
// Over a scripted DevTools socket, with no browser and on a simulated clock,
// the page reports its state and then goes quiet. A predicate it never makes
// true, an expression it never answers and an event it never sends each end
// when the budget has passed, with the caller's context still alive: the
// failure is a deadline's, names the budget, and for the predicate carries
// what the page last reported. A page that answers inside the budget is
// answered as before. And a budget the lane declared that cannot be read
// starts no browser.
func TestDeclaredBudgetEndsAWaitThePageNeverAnswers(t *testing.T) {
	// declared is the budget this test declares for its sockets, on the simulated clock.
	const declared = 90 * time.Second
	reported := `{"method":"Runtime.bindingCalled","params":{"name":"` + pageStateBinding + `","payload":"readyState complete, panels \"Could not load the workspace modules.\"","executionContextId":1}}`
	for _, wait := range []struct {
		name string
		page []string
		run  func(ctx context.Context, socket *webSocket) error
		want []string
	}{
		{"a predicate the page never makes true", []string{reported}, func(ctx context.Context, socket *webSocket) error {
			return (&Browser{socket: socket}).Eventually(ctx, "false")
		}, []string{"Could not load the workspace modules."}},
		{"an expression the page never answers", nil, func(ctx context.Context, socket *webSocket) error {
			return (&Browser{socket: socket}).Evaluate(ctx, "new Promise(() => {})", nil)
		}, nil},
		{"an event the page never sends", nil, func(ctx context.Context, socket *webSocket) error {
			return socket.awaitEvent(ctx, func(string, json.RawMessage) bool { return false })
		}, nil},
	} {
		t.Run(wait.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				socket, send := pageSide(t)
				socket.budget = declared
				send(wait.page...)
				err := wait.run(t.Context(), socket)
				if !errors.Is(err, context.DeadlineExceeded) || t.Context().Err() != nil {
					t.Fatalf("the wait ended with %v, its caller's context ended: %t; want the budget's deadline under a caller still waiting", err, t.Context().Err() != nil)
				}
				for _, want := range append(wait.want, declared.String()) {
					if !strings.Contains(err.Error(), want) {
						t.Fatalf("the wait ended with %v, want it to name %q", err, want)
					}
				}
			})
		})
	}
	t.Run("a page that answers inside the budget", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			socket, send := pageSide(t)
			socket.budget = declared
			send(`{"id":1,"result":{"result":{"value":true}}}`)
			if err := (&Browser{socket: socket}).Eventually(t.Context(), "true"); err != nil {
				t.Fatalf("a predicate the page answered true inside the budget = %v", err)
			}
		})
	})
	for _, unread := range []string{"soon", "-1s"} {
		t.Setenv(WaitBudgetEnvironment, unread)
		if _, err := Open(t.Context(), "no-browser", "about:blank"); err == nil || !strings.Contains(err.Error(), WaitBudgetEnvironment) {
			t.Fatalf("a browser opened under a declared budget of %q = %v, want it refused before anything is launched, naming the variable", unread, err)
		}
	}
}
