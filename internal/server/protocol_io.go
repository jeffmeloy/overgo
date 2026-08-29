package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sync"
	"time"
)

var errRequestTimeoutCause = errors.New("server: request timeout elapsed")

func beginSSE(response http.ResponseWriter) (http.Flusher, bool) {
	flusher, ok := response.(http.Flusher)
	if !ok {
		writeError(response, http.StatusInternalServerError, "server_error", "streaming is unavailable")
		return nil, false
	}
	response.Header().Set("Content-Type", "text/event-stream")
	response.Header().Set("Cache-Control", "no-cache")
	response.Header().Set("X-Accel-Buffering", "no")
	response.WriteHeader(http.StatusOK)
	return flusher, true
}

type sseEmitter struct {
	ctx     context.Context
	writer  io.Writer
	flusher http.Flusher
}

func newSSEEmitter(ctx context.Context, writer io.Writer, flusher http.Flusher) sseEmitter {
	return sseEmitter{ctx: ctx, writer: writer, flusher: flusher}
}

func (stream sseEmitter) write(value any) error {
	if err := writeSSE(stream.writer, value); err != nil {
		return err
	}
	stream.flusher.Flush()
	return stream.ctx.Err()
}

func (stream sseEmitter) named(name string, value any) error {
	if err := writeSSEEvent(stream.writer, name, value); err != nil {
		return err
	}
	stream.flusher.Flush()
	return stream.ctx.Err()
}

func (stream sseEmitter) done() error {
	if _, err := io.WriteString(stream.writer, "data: [DONE]\n\n"); err != nil {
		return err
	}
	stream.flusher.Flush()
	return stream.ctx.Err()
}

type synchronizedSSE struct {
	mu      sync.Mutex
	writer  io.Writer
	flusher http.Flusher
	err     error
}

func newSynchronizedSSE(writer io.Writer, flusher http.Flusher) *synchronizedSSE {
	return &synchronizedSSE{writer: writer, flusher: flusher}
}

func (stream *synchronizedSSE) write(value any) error {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if stream.err != nil {
		return stream.err
	}
	stream.err = writeSSE(stream.writer, value)
	if stream.err == nil {
		stream.flusher.Flush()
	}
	return stream.err
}

func (stream *synchronizedSSE) ping() error {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if stream.err != nil {
		return stream.err
	}
	_, stream.err = io.WriteString(stream.writer, ":\n\n")
	if stream.err == nil {
		stream.flusher.Flush()
	}
	return stream.err
}

func (stream *synchronizedSSE) startHeartbeat(ctx context.Context, interval time.Duration) func() {
	if interval <= 0 {
		return func() {}
	}
	stop := make(chan struct{})
	stopped := make(chan struct{})
	stopHeartbeat := sync.OnceFunc(func() { close(stop) })
	stopAfterContext := context.AfterFunc(ctx, stopHeartbeat)
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				if stream.ping() != nil {
					return
				}
			}
		}
	}()
	return func() {
		stopAfterContext()
		stopHeartbeat()
		<-stopped
	}
}

func writeGenerationError(response http.ResponseWriter, err error) {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, errRequestTimeoutCause) {
		writeError(response, http.StatusRequestTimeout, "request_cancelled", err.Error())
		return
	}
	writeError(response, http.StatusInternalServerError, "generation_error", err.Error())
}

func writeJSON(response http.ResponseWriter, status int, value any) {
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(status)
	_ = json.NewEncoder(response).Encode(value)
}

func writeSSE(response io.Writer, value any) error {
	return writeSSEEvent(response, "", value)
}

func writeSSEEvent(response io.Writer, event string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if event != "" {
		if _, err = io.WriteString(response, "event: "+event+"\n"); err != nil {
			return err
		}
	}
	if _, err = io.WriteString(response, "data: "); err != nil {
		return err
	}
	if _, err = response.Write(data); err != nil {
		return err
	}
	_, err = io.WriteString(response, "\n\n")
	return err
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type apiErrorEnvelope struct {
	Error apiError `json:"error"`
}

func errorEnvelope(kind, message string) apiErrorEnvelope {
	return apiErrorEnvelope{Error: apiError{Message: message, Type: kind}}
}

func writeError(response http.ResponseWriter, status int, kind, message string) {
	writeJSON(response, status, errorEnvelope(kind, message))
}
