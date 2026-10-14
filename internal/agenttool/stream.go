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
	// HTTPJSONStreamMediaType is the newline-delimited JSON stream transport type.
	HTTPJSONStreamMediaType = "application/x-ndjson"
	// StreamTerminalMediaType identifies encoded stream-terminal documents.
	StreamTerminalMediaType = "application/vnd.overgo.agent-tool-stream-terminal+json"
	// StreamTerminalSchema identifies the exact stored stream-terminal schema.
	StreamTerminalSchema = "overgo/agent-tool-stream-terminal/v1"
	// StreamEventSchema identifies one governed stream event envelope.
	StreamEventSchema = "overgo.agent-tool-stream-event.v1"

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
	// StreamCompleted reports a stream that delivered every event and its terminal.
	StreamCompleted StreamState = "completed"
	// StreamFailed reports a stream that ended with a bounded error payload.
	StreamFailed StreamState = "failed"
	// StreamCancelled reports a stream ended by its consumer before the terminal.
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
	Error      string      `json:"error,omitzero"`
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
// transport body, and terminal evidence lifecycle.
type InvocationStream struct {
	ctx        context.Context
	cancel     context.CancelFunc
	body       io.ReadCloser
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

// invoke returns a stream's events as one JSON array in sequence order.
//   - the stream's bytes are bounded by artifact.MaxContentBytes, the bound boundedResult holds every transport's result to
//   - the array's brackets and commas are not stream bytes, so boundedResult refuses an array they carry past that bound
//   - a stream that fails, passes its admission bound or is cancelled is refused with its state, its counts and its error
//   - no partial array is returned
//   - each event reaches the context's output observer as it arrives, one line per event
//   - the terminal is not stored because the coordinator stores the array, and a refusal carries the terminal's state, counts and error
func (adapter *httpJSONStreamAdapter) invoke(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	stream, err := adapter.openStream(ctx, manual, arguments, StreamPolicy{
		MaxEvents: streamMaxEvents, MaxEventBytes: artifact.MaxContentBytes, MaxBytes: artifact.MaxContentBytes,
	})
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	observe, _ := ctx.Value(outputObserverKey{}).(func([]byte))
	collected := json.RawMessage{'['}
	var received uint64
	for {
		event, err := stream.Receive(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("stream cancelled after %d events: %w", received, context.Cause(ctx))
		}
		if received != 0 {
			collected = append(collected, ',')
		}
		collected, received = append(collected, event.Value...), event.Sequence
		if observe != nil {
			observe(append(event.Value, '\n'))
		}
	}
	terminal, err := stream.Terminal()
	switch {
	case err != nil:
		return nil, err
	case terminal.State == StreamCancelled:
		return nil, fmt.Errorf("stream cancelled after %d events: %w", terminal.Events, context.Cause(ctx))
	case terminal.State != StreamCompleted:
		return nil, fmt.Errorf("stream %s after %d events of %d bytes: %s", terminal.State, terminal.Events, terminal.Bytes, terminal.Error)
	}
	return append(collected, ']'), nil
}

// openStream admits one stream and starts a bounded backpressure pump.
//   - the caller holds the manual's entry for the stream's whole life (Executor.InvokeWithEffect)
//   - the caller has validated the arguments against the manual
func (adapter *httpJSONStreamAdapter) openStream(
	ctx context.Context,
	manual Manual,
	arguments json.RawMessage,
	policy StreamPolicy,
) (*InvocationStream, error) {
	if err := policy.validate(); err != nil {
		return nil, err
	}
	invocation, err := artifact.JSONID(artifact.KindRun, struct {
		Manual    artifact.ID     `json:"manual"`
		Arguments json.RawMessage `json:"arguments"`
		Policy    StreamPolicy    `json:"policy"`
	}{Manual: manual.ID, Arguments: arguments, Policy: policy})
	if err != nil {
		return nil, err
	}
	bounded, cancelCause := context.WithCancelCause(ctx)
	cancel := func() { cancelCause(context.Canceled) }
	body, err := adapter.open(bounded, manual, arguments)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("agent tool: %q stream failed: %w", manual.Name, err)
	}
	stream := &InvocationStream{
		ctx: bounded, cancel: cancel, body: body, policy: policy,
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
			stream.finish(StreamFailed, count, total, fmt.Errorf(
				"stream admission bound exceeded at event %d with %d bytes, and the policy admits %d events of %d bytes",
				nextCount, nextTotal, stream.policy.MaxEvents, stream.policy.MaxBytes))
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
