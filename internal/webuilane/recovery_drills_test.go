package webuilane

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TestRecoveryDrills replays the lane failure where one browser test waited
// twenty minutes on a predicate the page never met and the lane ended with
// no page state. Over a scripted DevTools socket, with no browser, the page
// reports its state through the wait's binding, then the wait ends without
// its predicate: the caller's context ends it, or the page crashes or
// detaches. The recovery is a failure that carries what the page last
// reported; that text in the lane's log is the signal, where before there
// was only go test's timeout.
func TestRecoveryDrills(t *testing.T) {
	t.Parallel()
	state := `readyState complete, panels \"Could not load the workspace modules.\", last error \"TypeError: modules is undefined\"`
	reported := `{"method":"Runtime.bindingCalled","params":{"name":"` + pageStateBinding + `","payload":"` + state + `","executionContextId":1}}`
	stopped := errors.New("the lane stopped the wait")
	for _, drill := range []struct {
		name string
		// page is what the page sends while the wait runs; stop ends the
		// wait through the caller's context once the lane has read it.
		page []string
		stop bool
		want []string
	}{
		{
			name: "the caller ends a wait the page never answers", page: []string{reported}, stop: true,
			want: []string{"did not become true", stopped.Error(), "Could not load the workspace modules.", "TypeError: modules is undefined"},
		},
		{
			name: "the page crashes during the wait", page: []string{reported, `{"method":"Inspector.targetCrashed","params":{}}`},
			want: []string{"Inspector.targetCrashed", "readyState complete"},
		},
		{
			name: "a page that reported nothing says so", page: []string{`{"method":"Inspector.detached","params":{"reason":"target_closed"}}`},
			want: []string{"Inspector.detached", "the page last reported nothing"},
		},
	} {
		t.Run(drill.name, func(t *testing.T) {
			t.Parallel()
			socket, send := pageSide(t)
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			read := send(drill.page...)
			if drill.stop {
				go func() { <-read; cancel(stopped) }()
			}
			err := (&Browser{socket: socket}).Eventually(ctx, "false")
			if err == nil || drill.stop != errors.Is(err, stopped) {
				t.Fatalf("wait ended with %v; want a failure, ended by the lane only when it stopped the wait", err)
			}
			for _, want := range drill.want {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("wait ended with %v, want it to name %q", err, want)
				}
			}
		})
	}
}
