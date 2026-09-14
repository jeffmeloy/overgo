package modelswap

import (
	"bytes"
	"context"
	"errors"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestProxyClientCancellation(t *testing.T) {
	previous := log.Writer()
	var messages bytes.Buffer
	log.SetOutput(&messages)
	t.Cleanup(func() { log.SetOutput(previous) })
	cancelled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	expired, release := context.WithTimeoutCause(t.Context(), 0, context.DeadlineExceeded)
	defer release()
	upstream := errors.New("fixture upstream failure")
	for _, test := range []struct {
		name      string
		ctx       context.Context
		err       error
		cancelled bool
	}{
		{"client cancelled", cancelled, context.Canceled, true},
		{"upstream cancelled", t.Context(), context.Canceled, false},
		{"request deadline", expired, context.DeadlineExceeded, false},
		{"upstream failure", t.Context(), upstream, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			messages.Reset()
			response := httptest.NewRecorder()
			response.Code = 0
			request := httptest.NewRequestWithContext(test.ctx, http.MethodGet, "/generation/capabilities", nil)
			writeProxyError(response, request, test.err)
			if test.cancelled {
				if response.Code != 0 || response.Body.Len() != 0 || messages.Len() != 0 {
					t.Fatalf("cancelled client emitted a gateway failure: %d %q %q", response.Code, response.Body.String(), messages.String())
				}
				return
			}
			if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "Bad Gateway") || !strings.Contains(messages.String(), test.err.Error()) {
				t.Fatalf("live failure lost status/diagnostic: %d %q %q", response.Code, response.Body.String(), messages.String())
			}
		})
	}
}
