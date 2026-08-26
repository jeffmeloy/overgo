package agenttool

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"slices"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
	"overgo/internal/textcheck"
)

const (
	HTTPJSONStreamMediaType = "application/x-ndjson"
	StreamTerminalMediaType = "application/vnd.overgo.agent-tool-stream-terminal+json"
	StreamTerminalSchema    = "overgo/agent-tool-stream-terminal/v1"
	StreamEventSchema       = "overgo.agent-tool-stream-event.v1"

	streamMaxEvents       = 65536
	streamMaxErrorBytes   = 4096
	streamEventQueueDepth = 1
)

// StreamPolicy is caller-owned admission authority for one invocation.
type StreamPolicy struct {
	MaxEvents     uint32 `json:"max_events"`
	MaxEventBytes uint64 `json:"max_event_bytes"`
	MaxBytes      uint64 `json:"max_bytes"`
}

// StreamEvent is one ordered partial result bound to its invocation and tool.
type StreamEvent struct {
	Invocation artifact.ID     `json:"invocation"`
	Manual     artifact.ID     `json:"manual"`
	Sequence   uint64          `json:"sequence"`
	Value      json.RawMessage `json:"value"`
}

// ArtifactContent returns one durable partial-result boundary.
func (event StreamEvent) ArtifactContent() (artifact.Content, error) {
	if event.Invocation.Kind() != artifact.KindRun || event.Manual.Kind() != artifact.KindRecipe ||
		event.Sequence == 0 || !json.Valid(event.Value) {
		return artifact.Content{}, errors.New("agent tool: invalid stream event")
	}
	return artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, StreamEventSchema), event)
}

// StreamState is the exact terminal disposition of one admitted stream.
type StreamState string

const (
	StreamCompleted StreamState = "completed"
	StreamFailed    StreamState = "failed"
	StreamCancelled StreamState = "cancelled"
)

// StreamTerminal closes one invocation exactly once.
type StreamTerminal struct {
	Version    uint16      `json:"version"`
	Invocation artifact.ID `json:"invocation"`
	Manual     artifact.ID `json:"manual"`
	State      StreamState `json:"state"`
	Events     uint32      `json:"events"`
	Bytes      uint64      `json:"bytes"`
	Error      string      `json:"error,omitempty"`
	ID         artifact.ID `json:"-"`
}

var streamTerminalCodec = artifact.JSONDocumentCodec(
	"agent tool stream terminal", artifact.KindEvidence, StreamTerminalMediaType, StreamTerminalSchema,
	canonicalizeStreamTerminal,
	func(value StreamTerminal) artifact.ID { return value.ID },
	func(value *StreamTerminal, id artifact.ID) { value.ID = id },
	func(value StreamTerminal) StreamTerminal { return value },
)

// ArtifactContent returns the terminal's canonical evidence bytes.
func (terminal StreamTerminal) ArtifactContent() (artifact.Content, error) {
	return streamTerminalCodec.Content(terminal)
}

// InvocationStream receives a bounded sequence and owns its cancellation,
// transport body, per-manual entry lease, and terminal evidence lifecycle.
type InvocationStream struct {
	ctx        context.Context
	cancel     context.CancelFunc
	body       io.ReadCloser
	release    func()
	policy     StreamPolicy
	invocation artifact.ID
	manual     artifact.ID

	events   chan StreamEvent
	done     chan struct{}
	mu       sync.Mutex
	closed   bool
	final    StreamTerminal
	finalErr error
}

type streamTransportAdapter interface {
	open(context.Context, Manual, json.RawMessage) (io.ReadCloser, error)
}

// OpenStream admits one stream and starts a bounded backpressure pump.
func (executor *Executor) OpenStream(
	ctx context.Context,
	manual Manual,
	arguments json.RawMessage,
	policy StreamPolicy,
) (*InvocationStream, error) {
	if executor == nil || ctx == nil {
		return nil, errors.New("agent tool: stream executor or context is absent")
	}
	if err := validateArguments(manual, arguments); err != nil {
		return nil, err
	}
	if err := policy.validate(); err != nil {
		return nil, err
	}
	adapter, found := executor.adapters[manual.Transport.Kind]
	streamAdapter, supported := adapter.(streamTransportAdapter)
	if !found || !supported {
		return nil, fmt.Errorf("agent tool: transport %q is not streamable", manual.Transport.Kind)
	}
	invocation, err := artifact.JSONID(artifact.KindRun, struct {
		Manual    artifact.ID     `json:"manual"`
		Arguments json.RawMessage `json:"arguments"`
		Policy    StreamPolicy    `json:"policy"`
	}{Manual: manual.ID, Arguments: arguments, Policy: policy})
	if err != nil {
		return nil, err
	}
	bounded, cancel := context.WithTimeout(ctx, invokeTimeout)
	entry := executor.manualEntry(manual.Name)
	select {
	case entry <- struct{}{}:
	case <-bounded.Done():
		cancel()
		return nil, fmt.Errorf("agent tool: %q timed out waiting for stream entry: %w", manual.Name, bounded.Err())
	}
	release := func() { <-entry }
	body, err := streamAdapter.open(bounded, manual, arguments)
	if err != nil {
		release()
		cancel()
		return nil, fmt.Errorf("agent tool: %q stream failed: %w", manual.Name, err)
	}
	stream := &InvocationStream{
		ctx: bounded, cancel: cancel, body: body, release: release, policy: policy,
		invocation: invocation, manual: manual.ID,
		events: make(chan StreamEvent, streamEventQueueDepth), done: make(chan struct{}),
	}
	go stream.pump()
	return stream, nil
}

// Receive waits for one event, EOF after completion, or caller cancellation.
// Cancelling a receive cancels the whole invocation so there is one lifecycle.
func (stream *InvocationStream) Receive(ctx context.Context) (StreamEvent, error) {
	if stream == nil || ctx == nil {
		return StreamEvent{}, errors.New("agent tool: stream or receive context is absent")
	}
	if err := ctx.Err(); err != nil {
		stream.cancelInvocation()
		return StreamEvent{}, err
	}
	select {
	case event, open := <-stream.events:
		if !open {
			return StreamEvent{}, io.EOF
		}
		event.Value = slices.Clone(event.Value)
		return event, nil
	case <-ctx.Done():
		stream.cancelInvocation()
		return StreamEvent{}, ctx.Err()
	}
}

// Close cancels unfinished work and waits until the terminal is exact.
func (stream *InvocationStream) Close() error {
	if stream == nil {
		return nil
	}
	stream.cancelInvocation()
	<-stream.done
	return nil
}

// Terminal returns the exact terminal evidence after the stream ends.
func (stream *InvocationStream) Terminal() (StreamTerminal, error) {
	if stream == nil {
		return StreamTerminal{}, errors.New("agent tool: stream is absent")
	}
	select {
	case <-stream.done:
		stream.mu.Lock()
		defer stream.mu.Unlock()
		return stream.final, stream.finalErr
	default:
		return StreamTerminal{}, errors.New("agent tool: stream is still running")
	}
}

func (stream *InvocationStream) pump() {
	defer func() {
		_ = stream.body.Close()
		stream.cancel()
		stream.release()
		close(stream.done)
		close(stream.events)
	}()
	scanner := bufio.NewScanner(stream.body)
	scanner.Buffer(make([]byte, min(int(stream.policy.MaxEventBytes), bufio.MaxScanTokenSize)), int(stream.policy.MaxEventBytes)+1)
	var count uint32
	var total uint64
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var value json.RawMessage
		if err := strictjson.DecodeBytes(line, &value); err != nil {
			stream.finish(StreamFailed, count, total, fmt.Errorf("invalid stream JSON: %w", err))
			return
		}
		nextCount, nextTotal := count+1, total+uint64(len(line))
		if nextCount > stream.policy.MaxEvents || nextTotal > stream.policy.MaxBytes {
			stream.finish(StreamFailed, count, total, errors.New("stream admission bound exceeded"))
			return
		}
		event := StreamEvent{
			Invocation: stream.invocation, Manual: stream.manual,
			Sequence: uint64(nextCount), Value: slices.Clone(value),
		}
		count, total = nextCount, nextTotal
		select {
		case stream.events <- event:
		case <-stream.ctx.Done():
			stream.finish(StreamCancelled, count, total, nil)
			return
		}
	}
	if err := scanner.Err(); err != nil && stream.ctx.Err() == nil {
		stream.finish(StreamFailed, count, total, err)
		return
	}
	if stream.ctx.Err() != nil {
		stream.finish(StreamCancelled, count, total, nil)
		return
	}
	stream.finish(StreamCompleted, count, total, nil)
}

func (stream *InvocationStream) cancelInvocation() {
	stream.mu.Lock()
	cancel := !stream.closed && stream.final.ID == (artifact.ID{})
	if cancel {
		stream.closed = true
	}
	stream.mu.Unlock()
	if cancel {
		stream.cancel()
		_ = stream.body.Close()
	}
}

func (stream *InvocationStream) finish(state StreamState, events uint32, bytes uint64, failure error) {
	message := ""
	if failure != nil {
		message = failure.Error()
		if !textcheck.Bounded(message, streamMaxErrorBytes, "\x00") {
			message = "stream failure detail exceeded the bound"
		}
	}
	terminal, err := streamTerminalCodec.New(StreamTerminal{
		Version: artifact.InitialDocumentVersion, Invocation: stream.invocation,
		Manual: stream.manual, State: state, Events: events, Bytes: bytes, Error: message,
	})
	stream.mu.Lock()
	stream.final, stream.finalErr = terminal, err
	stream.mu.Unlock()
}

func (policy StreamPolicy) validate() error {
	if policy.MaxEvents == 0 || policy.MaxEvents > streamMaxEvents ||
		policy.MaxEventBytes == 0 || policy.MaxEventBytes > artifact.MaxContentBytes ||
		policy.MaxBytes < policy.MaxEventBytes || policy.MaxBytes > artifact.MaxContentBytes {
		return errors.New("agent tool: invalid stream admission policy")
	}
	return nil
}

func canonicalizeStreamTerminal(terminal *StreamTerminal) error {
	if terminal == nil || terminal.Version != artifact.InitialDocumentVersion ||
		terminal.Invocation.Kind() != artifact.KindRun || terminal.Manual.Kind() != artifact.KindRecipe {
		return errors.New("agent tool: invalid stream terminal authority")
	}
	switch terminal.State {
	case StreamCompleted, StreamCancelled:
		if terminal.Error != "" {
			return errors.New("agent tool: successful or cancelled stream carries an error")
		}
	case StreamFailed:
		if terminal.Error == "" || !textcheck.Bounded(terminal.Error, streamMaxErrorBytes, "\x00") {
			return errors.New("agent tool: failed stream lacks bounded error evidence")
		}
	default:
		return errors.New("agent tool: invalid stream terminal state")
	}
	return nil
}

func (adapter *httpJSONStreamAdapter) open(
	ctx context.Context,
	manual Manual,
	arguments json.RawMessage,
) (io.ReadCloser, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, manual.Transport.URL, bytes.NewReader(arguments))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", ManualMediaType)
	request.Header.Set("Accept", HTTPJSONStreamMediaType)
	response, err := adapter.client.Do(request)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		response.Body.Close()
		return nil, fmt.Errorf("stream endpoint returned status %d", response.StatusCode)
	}
	mediaType, _, err := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if err != nil || mediaType != HTTPJSONStreamMediaType {
		response.Body.Close()
		return nil, errors.New("stream endpoint returned an incompatible media type")
	}
	return response.Body, nil
}
