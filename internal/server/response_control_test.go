package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/tokenizer"
)

// Generation is held at an observable execution boundary, independently of
// the HTTP connection. Only explicit cancellation or the test releases it.
type responseControlGenerator struct {
	*recipeInspectorGenerator
	starts             chan context.Context
	release            chan error
	calls              atomic.Int32
	prefix             string
	opaqueCancellation bool
}

func (g *responseControlGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	g.calls.Add(1)
	if g.prefix != "" && options.OnToken != nil {
		if err := options.OnToken(inference.TokenEvent{ID: 30, Piece: g.prefix}); err != nil {
			return nil, "", err
		}
	}
	select {
	case g.starts <- ctx:
	case <-ctx.Done():
		return nil, "", ctx.Err()
	}
	select {
	case err := <-g.release:
		if err != nil {
			return nil, "", err
		}
		return g.recipeInspectorGenerator.Generate(ctx, prompt, options)
	case <-ctx.Done():
		if g.opaqueCancellation {
			return nil, "", errors.New(ctx.Err().Error())
		}
		return nil, "", ctx.Err()
	}
}

type disconnectedResponseWriter struct {
	recorder *httptest.ResponseRecorder
	event    string
	cancel   context.CancelFunc
	dropped  bool
}

type stalledResponseWriter struct {
	*httptest.ResponseRecorder
	blocked     chan struct{}
	interrupted chan struct{}
	unblock     func()
}

func (w *stalledResponseWriter) WriteString(value string) (int, error) {
	return w.Write([]byte(value))
}

func (w *stalledResponseWriter) Write(value []byte) (int, error) {
	if strings.Contains(string(value), "event: response.output_text.delta\n") {
		close(w.blocked)
		<-w.interrupted
		return 0, io.ErrClosedPipe
	}
	return w.ResponseRecorder.Write(value)
}

func (w *stalledResponseWriter) SetWriteDeadline(time.Time) error {
	w.unblock()
	return nil
}

func TestStoredResponseStopInterruptsStalledWrite(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	interrupted := make(chan struct{})
	writer := &stalledResponseWriter{ResponseRecorder: httptest.NewRecorder(), blocked: make(chan struct{}), interrupted: interrupted,
		unblock: sync.OnceFunc(func() { close(interrupted) })}
	defer writer.SetWriteDeadline(time.Time{})
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(writer, httptest.NewRequest(http.MethodPost, "/v1/responses",
			strings.NewReader(`{"input":"Slow connection","stream":true,"store":true,"max_output_tokens":2}`)))
		close(done)
	}()
	select {
	case <-writer.blocked:
	case <-done:
		t.Fatalf("response did not reach the stalled write: %s", writer.Body)
	}
	var listed conversationListResponse
	list := serveTestRequest(handler, http.MethodGet, "/interactions", "")
	if err := json.Unmarshal(list.Body.Bytes(), &listed); err != nil || len(listed.Conversations) != 1 {
		t.Fatalf("reserved prompt = %s (%v)", list.Body, err)
	}
	id := listed.Conversations[0].Latest
	cancelled := serveTestRequest(handler, http.MethodPost, "/interactions/cancel", `{"response":"`+id+`","model":"test-model"}`)
	if cancelled.Code != http.StatusOK {
		t.Fatalf("cancel = %d %s", cancelled.Code, cancelled.Body)
	}
	<-done
	follow := serveTestRequest(handler, http.MethodGet, "/interactions/follow?response="+id, "")
	if !strings.Contains(follow.Body.String(), "event: response.cancelled") || strings.Contains(follow.Body.String(), "event: response.completed") {
		t.Fatalf("stalled response recovery = %s", follow.Body)
	}
	if err := handler.Close(); err != nil {
		t.Fatalf("serving slot stayed leased after Stop: %v", err)
	}
}

func (w *disconnectedResponseWriter) Header() http.Header  { return w.recorder.Header() }
func (w *disconnectedResponseWriter) WriteHeader(code int) { w.recorder.WriteHeader(code) }
func (w *disconnectedResponseWriter) Flush()               { w.recorder.Flush() }
func (w *disconnectedResponseWriter) Write(p []byte) (int, error) {
	if strings.Contains(string(p), "event: "+w.event+"\n") {
		w.dropped = true
		w.cancel()
	}
	if w.dropped {
		return 0, io.ErrClosedPipe
	}
	return w.recorder.Write(p)
}

func TestStoredResponseDisconnectFinalization(t *testing.T) {
	for _, event := range []string{"response.created", "response.output_item.added", "response.output_text.done"} {
		t.Run(event, func(t *testing.T) {
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
			ctx, cancelCause := context.WithCancelCause(t.Context())
			cancel := func() { cancelCause(io.ErrClosedPipe) }
			defer cancel()
			writer := &disconnectedResponseWriter{recorder: httptest.NewRecorder(), event: event, cancel: cancel}
			request := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"Keep my prompt","stream":true,"store":true,"max_output_tokens":2}`))
			handler.ServeHTTP(writer, request)
			if !writer.dropped {
				t.Fatalf("disconnect boundary was not exercised: %d %s", writer.recorder.Code, writer.recorder.Body)
			}
			listed := serveTestRequest(handler, http.MethodGet, "/interactions", "")
			var list conversationListResponse
			if err := json.Unmarshal(listed.Body.Bytes(), &list); err != nil || len(list.Conversations) != 1 {
				t.Fatalf("durable conversations: %s (%v)", listed.Body, err)
			}
			id := list.Conversations[0].Latest
			messages, _, found := handler.loadResponseInteraction(t.Context(), id)
			if !found || len(messages) != 2 || messages[0].Content != "Keep my prompt" || messages[1].Content != "AB" {
				t.Fatalf("disconnected transcript = %+v, found=%v", messages, found)
			}
			turn, found := handler.inflight.lookup(id)
			if !found {
				t.Fatal("response was not registered before delivery")
			}
			text, done, final, failed, _ := turn.snapshot()
			if !done || text != "AB" || final.Status != "completed" || failed != "" {
				t.Fatalf("terminal = %q %v %+v %q", text, done, final, failed)
			}
			if err := handler.Close(); err != nil {
				t.Fatal(err)
			}
			restarted := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
			follow := serveTestRequest(restarted, http.MethodGet, "/interactions/follow?response="+id, "")
			if !strings.Contains(follow.Body.String(), "event: response.completed") || strings.Count(follow.Body.String(), `"delta":"AB"`) != 1 {
				t.Fatalf("durable replay = %s", follow.Body)
			}
		})
	}
}

func TestStoredResponseTerminalControl(t *testing.T) {
	for _, action := range []string{"cancel", "opaque cancel", "shutdown", "failure"} {
		t.Run(action, func(t *testing.T) {
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			generator := &responseControlGenerator{
				recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}),
				starts:                   make(chan context.Context), release: make(chan error), prefix: "Partial answer",
				opaqueCancellation: action == "opaque cancel",
			}
			handler := newTestHandlerForRepository(t, store, generator)
			done := make(chan *httptest.ResponseRecorder)
			go func() {
				done <- serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"input":"Keep this failed prompt","stream":true,"store":true,"max_output_tokens":2}`)
			}()
			var ctx context.Context
			select {
			case ctx = <-generator.starts:
			case rejected := <-done:
				t.Fatalf("response did not start: %d %s", rejected.Code, rejected.Body)
			}
			listed := serveTestRequest(handler, http.MethodGet, "/interactions", "")
			var list conversationListResponse
			if err := json.Unmarshal(listed.Body.Bytes(), &list); err != nil || len(list.Conversations) != 1 {
				t.Fatalf("initial prompt not durable: %s (%v)", listed.Body, err)
			}
			id := list.Conversations[0].Latest
			wrong := serveTestRequest(handler, http.MethodPost, "/interactions/cancel", `{"response":"`+id+`","model":"another-model"}`)
			if wrong.Code != http.StatusNotFound {
				t.Fatalf("wrong-model cancellation = %d", wrong.Code)
			}
			select {
			case <-ctx.Done():
				t.Fatal("wrong model cancelled execution")
			default:
			}
			pending := serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"input":"too early","previous_response_id":"`+id+`"}`)
			if pending.Code != http.StatusConflict {
				t.Fatalf("pending continuation = %d %s", pending.Code, pending.Body)
			}
			status := "cancelled"
			switch action {
			case "cancel", "opaque cancel":
				for range 2 {
					cancelled := serveTestRequest(handler, http.MethodPost, "/interactions/cancel", `{"response":"`+id+`","model":"test-model"}`)
					if cancelled.Code != http.StatusOK {
						t.Fatalf("cancel = %d %s", cancelled.Code, cancelled.Body)
					}
				}
				<-ctx.Done()
			case "shutdown":
				if err := handler.Close(); err != nil {
					t.Fatal(err)
				}
				<-ctx.Done()
			case "failure":
				status = "failed"
				generator.release <- errors.New("controlled generation failure")
			}
			result := <-done
			if !strings.Contains(result.Body.String(), "event: response."+status) || strings.Contains(result.Body.String(), "event: response.completed") {
				t.Fatalf("terminal stream = %s", result.Body)
			}
			if err := handler.Close(); err != nil {
				t.Fatal(err)
			}
			restarted := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
			follow := serveTestRequest(restarted, http.MethodGet, "/interactions/follow?response="+id, "")
			if !strings.Contains(follow.Body.String(), "event: response."+status) || !strings.Contains(follow.Body.String(), "Partial answer") || strings.Contains(follow.Body.String(), "event: response.completed") {
				t.Fatalf("terminal state lost on restart: %s", follow.Body)
			}
			messages, _, found := restarted.loadResponseInteraction(t.Context(), id)
			if !found || len(messages) != 2 || messages[0].Content != "Keep this failed prompt" {
				t.Fatalf("failed prompt lost: %+v", messages)
			}
			again := serveTestRequest(restarted, http.MethodPost, "/interactions/cancel", `{"response":"`+id+`","model":"test-model"}`)
			if again.Code != http.StatusOK || !strings.Contains(again.Body.String(), `"status":"`+status+`"`) {
				t.Fatalf("terminal cancellation retry = %d %s", again.Code, again.Body)
			}
			if generator.calls.Load() != 1 {
				t.Fatal("recovery generated another response")
			}
		})
	}
}

func TestStoredResponseUnconfirmedRestart(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
	if err := handler.publishResponseInteraction(t.Context(), "resp_9", artifact.ID{},
		[]inference.ChatMessage{{Role: inference.ChatRoleUser, Content: "private interrupted prompt"}}, runrecord.OutcomeInconclusive); err != nil {
		t.Fatal(err)
	}
	if err := handler.Close(); err != nil {
		t.Fatal(err)
	}
	restarted := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
	if restarted.nextID.Load() < 9 {
		t.Fatal("reserved response identifier was reused")
	}
	follow := serveTestRequest(restarted, http.MethodGet, "/interactions/follow?response=resp_9", "")
	if !strings.Contains(follow.Body.String(), "event: response.failed") || strings.Contains(follow.Body.String(), "private interrupted prompt") || strings.Contains(follow.Body.String(), "event: response.completed") {
		t.Fatalf("unconfirmed replay = %s", follow.Body)
	}
	messages := serveTestRequest(restarted, http.MethodGet, "/interactions/messages?response=resp_9", "")
	if !strings.Contains(messages.Body.String(), "private interrupted prompt") || !strings.Contains(messages.Body.String(), `"status":"failed"`) {
		t.Fatalf("unconfirmed prompt not recoverable: %s", messages.Body)
	}
}
