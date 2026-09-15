package modelswap

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestProxyResponseCancellation(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		ctx, cancel := context.WithCancelCause(t.Context())
		reader, writer := io.Pipe()
		body := &proxyResponseBody{ReadCloser: reader, ctx: ctx}
		done := make(chan error, 1)
		go func() {
			buffer := make([]byte, len("chunk"))
			n, err := body.Read(buffer)
			if err != nil || string(buffer[:n]) != "chunk" {
				done <- errors.New("streamed payload changed")
				return
			}
			_, err = body.Read(buffer)
			done <- err
		}()
		if _, err := writer.Write([]byte("chunk")); err != nil {
			t.Fatal(err)
		}
		if cancelled {
			cancel(context.Canceled)
		}
		_ = writer.CloseWithError(net.ErrClosed)
		err := <-done
		want := net.ErrClosed
		if cancelled {
			want = context.Canceled
		}
		if !errors.Is(err, want) {
			t.Fatalf("cancelled=%v: got %v, want %v", cancelled, err, want)
		}
		_ = body.Close()
		cancel(context.Canceled)
	}
}

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
