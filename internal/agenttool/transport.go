package agenttool

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/processcontrol"
	"overgo/internal/strictjson"
)

// Builtin is one in-process tool implementation: strict JSON in,
// strict JSON out.
type Builtin func(context.Context, json.RawMessage) (json.RawMessage, error)

// transportAdapter is the one native invocation boundary. The executor owns
// an immutable catalog of these adapters; protocols never branch through the
// orchestration path and cannot change manual identity or admission policy.
type transportAdapter interface {
	invoke(context.Context, Manual, json.RawMessage) (json.RawMessage, error)
}

type builtinAdapter struct {
	mu              sync.RWMutex
	implementations map[string]Builtin
}
type httpAdapter struct{ client *http.Client }
type argvAdapter struct{}
type mcpHTTPAdapter struct{ client *http.Client }
type httpJSONStreamAdapter struct{ client *http.Client }

// Executor invokes manuals over their declared native transports.
// Entry is serialized per manual: no transport promises concurrency
// safety, so two steps naming one tool never enter it at once. The
// entry slot is a one-place channel, not a mutex, so waiting for it
// honors the invocation deadline -- a wedged first call cannot make a
// second call wait past its own bound.
type Executor struct {
	entries  [sha256.Size]chan struct{}
	adapters map[TransportKind]transportAdapter
}

// NewExecutor returns the serving executor: no builtins registered,
// redirects refused, and http endpoints that resolve to loopback,
// private, or link-local addresses refused at dial time. Every
// network-facing surface uses this constructor.
func NewExecutor() *Executor {
	return newExecutor(newTransportClient(false))
}

// NewOperatorExecutor returns the operator's executor: identical
// policy except that loopback and private endpoints are reachable,
// because an operator invoking local tooling from the CLI is not a
// server fetching on a client's behalf. Redirects stay refused.
func NewOperatorExecutor() *Executor {
	return newExecutor(newTransportClient(true))
}

func newExecutor(client *http.Client) *Executor {
	builtins := &builtinAdapter{implementations: map[string]Builtin{}}
	executor := &Executor{
		adapters: map[TransportKind]transportAdapter{
			TransportBuiltin:        builtins,
			TransportHTTP:           &httpAdapter{client: client},
			TransportArgv:           argvAdapter{},
			TransportMCPHTTP:        &mcpHTTPAdapter{client: client},
			TransportHTTPJSONStream: &httpJSONStreamAdapter{client: client},
		},
	}
	for index := range executor.entries {
		executor.entries[index] = make(chan struct{}, 1)
	}
	return executor
}

func (adapter *httpJSONStreamAdapter) invoke(context.Context, Manual, json.RawMessage) (json.RawMessage, error) {
	return nil, errors.New("stream transport requires OpenStream")
}

func (e *Executor) manualEntry(name string) chan struct{} {
	digest := sha256.Sum256([]byte(name))
	return e.entries[int(digest[0])%len(e.entries)]
}

// registerBuiltin binds one in-process implementation to a manual name.
func (e *Executor) registerBuiltin(name string, implementation Builtin) error {
	if !manualNamePattern.MatchString(name) || implementation == nil {
		return fmt.Errorf("agent tool: invalid builtin registration %q", name)
	}
	builtins, ok := e.adapters[TransportBuiltin].(*builtinAdapter)
	if !ok {
		return errors.New("agent tool: builtin adapter is absent")
	}
	builtins.mu.Lock()
	defer builtins.mu.Unlock()
	if _, exists := builtins.implementations[name]; exists {
		return fmt.Errorf("agent tool: builtin %q is already registered", name)
	}
	builtins.implementations[name] = implementation
	return nil
}

// Invoke validates arguments against the manual and executes it over
// its declared transport, returning strict JSON or a typed error.
func (e *Executor) Invoke(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	result, _, err := e.InvokeWithEffect(ctx, manual, arguments)
	return result, err
}

// InvokeWithEffect returns the executed result and the exact canonical effect.
func (e *Executor) InvokeWithEffect(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, InvocationEffect, error) {
	if ctx == nil {
		return nil, InvocationEffect{}, errors.New("agent tool: nil invoke context")
	}
	if _, err := DeriveInvocationEffect(manual, arguments, nil); err != nil {
		return nil, InvocationEffect{}, err
	}
	// The caller's context bounds the whole invocation, the wait for the
	// manual's entry lock included; the wait ends with the tool or with
	// the caller, whatever transport it rides.
	entry := e.manualEntry(manual.Name)
	select {
	case entry <- struct{}{}:
		defer func() { <-entry }()
	case <-ctx.Done():
		return nil, InvocationEffect{}, fmt.Errorf("agent tool: %q timed out waiting for entry: %w", manual.Name, context.Cause(ctx))
	}
	adapter, registered := e.adapters[manual.Transport.Kind]
	if !registered {
		return nil, InvocationEffect{}, fmt.Errorf("agent tool: transport %q has no registered adapter", manual.Transport.Kind)
	}
	result, err := adapter.invoke(ctx, manual, arguments)
	if err != nil {
		return nil, InvocationEffect{}, fmt.Errorf("agent tool: %q failed: %w", manual.Name, err)
	}
	result, err = boundedResult(manual.Name, result)
	if err != nil {
		return nil, InvocationEffect{}, err
	}
	effect, err := DeriveInvocationEffect(manual, arguments, result)
	return result, effect, err
}

func (adapter *builtinAdapter) invoke(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	adapter.mu.RLock()
	implementation, registered := adapter.implementations[manual.Name]
	adapter.mu.RUnlock()
	if !registered {
		return nil, fmt.Errorf("builtin %q is not registered", manual.Name)
	}
	return implementation(ctx, arguments)
}

func (adapter *httpAdapter) invoke(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, manual.Transport.URL, bytes.NewReader(arguments))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", ManualMediaType)
	response, err := adapter.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("agent tool: %q endpoint failed: %w", manual.Name, err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, artifact.MaxContentBytes+1))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("agent tool: %q endpoint returned status %d", manual.Name, response.StatusCode)
	}
	if !json.Valid(body) {
		return nil, fmt.Errorf("agent tool: %q endpoint returned non-JSON", manual.Name)
	}
	return body, nil
}

func (argvAdapter) invoke(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	bounded, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	stdout := argvOutput{cancel: func() { cancel(&OutputLimitError{RetainedBytes: artifact.MaxContentBytes}) }}
	receipt, err := processcontrol.Run(bounded, processcontrol.Command{
		Path:   manual.Transport.Program,
		Args:   manual.Transport.Args,
		Stdin:  bytes.NewReader(arguments),
		Stdout: &stdout,
	})
	if stdout.overflow {
		return nil, &OutputLimitError{Receipt: receipt, RetainedBytes: len(stdout.data)}
	}
	if err != nil {
		return nil, fmt.Errorf("agent tool: %q exited: %w", manual.Name, err)
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("agent tool: %q exited with status %d", manual.Name, receipt.ExitCode)
	}
	// Argv tools speak text; the result is the stdout text as one JSON
	// string so every transport returns strict JSON to the loop.
	encoded, err := json.Marshal(strings.TrimRight(string(stdout.data), "\r\n"))
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

// OutputLimitError reports an argv stdout overflow after the supervised tree
// and output pipes have reached their terminal state. No partial result is
// returned; Receipt retains the actual process outcome.
type OutputLimitError struct {
	Receipt       processcontrol.Receipt
	RetainedBytes int
}

// Error describes the overflow bound and recorded tree termination.
func (err *OutputLimitError) Error() string {
	return fmt.Sprintf("argv stdout exceeds %d bytes (retained %d; tree terminated=%t)", artifact.MaxContentBytes, err.RetainedBytes, err.Receipt.TreeTerminated)
}

// argvOutput retains at most the artifact bound, including backing capacity.
// Only the stdout drain writes it; Run joins that drain before it is inspected.
type argvOutput struct {
	data     []byte
	cancel   context.CancelFunc
	overflow bool
}

// Write retains a bounded prefix and cancels on overflow while allowing drain.
func (output *argvOutput) Write(data []byte) (int, error) {
	retained := min(len(data), artifact.MaxContentBytes-len(output.data))
	needed := len(output.data) + retained
	if needed > cap(output.data) {
		capacity := min(artifact.MaxContentBytes, max(needed, cap(output.data)+cap(output.data)))
		grown := make([]byte, len(output.data), capacity)
		copy(grown, output.data)
		output.data = grown
	}
	output.data = append(output.data, data[:retained]...)
	if retained < len(data) && !output.overflow {
		output.overflow = true
		output.cancel()
	}
	// Keep draining after cancellation so the supervisor can join both pipes.
	return len(data), nil
}

const (
	mcpInitializeMethod  = "initialize"
	mcpInitializedMethod = "notifications/initialized"
	mcpToolsCallMethod   = "tools/call"
	mcpSessionHeader     = "Mcp-Session-Id"
	mcpClientName        = "overgo"
	mcpClientVersion     = "0.1.1"
)

type mcpRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      string          `json:"id,omitzero"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type mcpRequestParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type mcpInitializeParams struct {
	ProtocolVersion string                `json:"protocolVersion"`
	Capabilities    mcpClientCapabilities `json:"capabilities"`
	ClientInfo      mcpClientInfo         `json:"clientInfo"`
}

type mcpClientCapabilities struct{}

type mcpClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type mcpInitializeResult struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    json.RawMessage `json:"capabilities"`
	ServerInfo      json.RawMessage `json:"serverInfo"`
	Instructions    string          `json:"instructions,omitzero"`
}

type mcpResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      string          `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *mcpError       `json:"error,omitempty"`
}

type mcpError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (adapter *mcpHTTPAdapter) invoke(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	digest := sha256.Sum256(append([]byte(manual.ID.String()+"\x00"), arguments...))
	requestIdentity := fmt.Sprintf("%x", digest[:])
	initializeParams, err := json.Marshal(mcpInitializeParams{
		ProtocolVersion: manual.Transport.Protocol,
		Capabilities:    mcpClientCapabilities{},
		ClientInfo:      mcpClientInfo{Name: mcpClientName, Version: mcpClientVersion},
	})
	if err != nil {
		return nil, err
	}
	initializeID := requestIdentity + ".initialize"
	initializePayload, err := json.Marshal(mcpRequest{
		JSONRPC: "2.0", ID: initializeID, Method: mcpInitializeMethod, Params: initializeParams,
	})
	if err != nil {
		return nil, err
	}
	initializeBody, headers, err := adapter.post(ctx, manual, "", initializePayload, http.StatusOK)
	if err != nil {
		return nil, err
	}
	initialize, err := decodeMCPResponse(initializeBody, initializeID)
	if err != nil {
		return nil, err
	}
	var negotiated mcpInitializeResult
	if err := strictjson.DecodeBytes(initialize, &negotiated); err != nil ||
		negotiated.ProtocolVersion != manual.Transport.Protocol ||
		!strictjson.HasValue(negotiated.Capabilities) || !strictjson.HasValue(negotiated.ServerInfo) {
		return nil, errors.Join(errors.New("mcp initialization authority differs"), err)
	}
	session := headers.Get(mcpSessionHeader)
	initializedPayload, err := json.Marshal(mcpRequest{JSONRPC: "2.0", Method: mcpInitializedMethod})
	if err != nil {
		return nil, err
	}
	if _, _, err := adapter.post(ctx, manual, session, initializedPayload, http.StatusAccepted); err != nil {
		return nil, err
	}
	callParams, err := json.Marshal(mcpRequestParams{Name: manual.Transport.Target, Arguments: arguments})
	if err != nil {
		return nil, err
	}
	callID := requestIdentity + ".call"
	callPayload, err := json.Marshal(mcpRequest{
		JSONRPC: "2.0", ID: callID, Method: mcpToolsCallMethod, Params: callParams,
	})
	if err != nil {
		return nil, err
	}
	body, _, err := adapter.post(ctx, manual, session, callPayload, http.StatusOK)
	if err != nil {
		return nil, err
	}
	return decodeMCPResponse(body, callID)
}

func (adapter *mcpHTTPAdapter) post(
	ctx context.Context,
	manual Manual,
	session string,
	payload []byte,
	wantStatus int,
) ([]byte, http.Header, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, manual.Transport.URL, bytes.NewReader(payload))
	if err != nil {
		return nil, nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	request.Header.Set("MCP-Protocol-Version", manual.Transport.Protocol)
	if session != "" {
		request.Header.Set(mcpSessionHeader, session)
	}
	response, err := adapter.client.Do(request)
	if err != nil {
		return nil, nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, artifact.MaxContentBytes+1))
	if err != nil {
		return nil, nil, err
	}
	if response.StatusCode != wantStatus {
		return nil, nil, fmt.Errorf("mcp endpoint returned status %d, want %d", response.StatusCode, wantStatus)
	}
	return body, response.Header.Clone(), nil
}

func decodeMCPResponse(body []byte, requestID string) (json.RawMessage, error) {
	var envelope mcpResponse
	if err := strictjson.DecodeBytes(body, &envelope); err != nil {
		return nil, fmt.Errorf("decode mcp response: %w", err)
	}
	if envelope.JSONRPC != "2.0" || envelope.ID != requestID {
		return nil, errors.New("mcp response identity differs")
	}
	if envelope.Error != nil {
		return nil, fmt.Errorf("mcp error %d: %s", envelope.Error.Code, envelope.Error.Message)
	}
	if !strictjson.HasValue(envelope.Result) {
		return nil, errors.New("mcp response result is absent")
	}
	return envelope.Result, nil
}

func boundedResult(name string, result json.RawMessage) (json.RawMessage, error) {
	if len(result) > artifact.MaxContentBytes {
		return nil, fmt.Errorf("agent tool: %q result exceeds the size bound", name)
	}
	if len(result) == 0 || !json.Valid(result) {
		return nil, fmt.Errorf("agent tool: %q returned non-JSON", name)
	}
	return append(json.RawMessage(nil), result...), nil
}

func validateArguments(manual Manual, arguments json.RawMessage) error {
	var supplied map[string]json.RawMessage
	if err := strictjson.DecodeBytes(arguments, &supplied); err != nil || supplied == nil {
		return fmt.Errorf("agent tool: %q arguments must be a strict JSON object", manual.Name)
	}
	declared := map[string]Field{}
	for _, field := range manual.Arguments {
		declared[field.Name] = field
		if _, present := supplied[field.Name]; field.Required && !present {
			return fmt.Errorf("agent tool: %q requires argument %q", manual.Name, field.Name)
		}
	}
	for name, value := range supplied {
		field, ok := declared[name]
		if !ok {
			return fmt.Errorf("agent tool: %q does not declare argument %q", manual.Name, name)
		}
		if err := checkFieldKind(field, value); err != nil {
			return fmt.Errorf("agent tool: %q argument %q: %w", manual.Name, name, err)
		}
	}
	return nil
}

func checkFieldKind(field Field, value json.RawMessage) error {
	trimmed := bytes.TrimSpace(value)
	if len(trimmed) == 0 {
		return errors.New("empty value")
	}
	switch field.Kind {
	case FieldString:
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return errors.New("expected a JSON string")
		}
	case FieldBoolean:
		var flag bool
		if err := json.Unmarshal(trimmed, &flag); err != nil {
			return errors.New("expected a JSON boolean")
		}
	case FieldNumber:
		var number float64
		if err := json.Unmarshal(trimmed, &number); err != nil {
			return errors.New("expected a JSON number")
		}
	case FieldInteger:
		var integer int64
		if err := json.Unmarshal(trimmed, &integer); err != nil {
			return errors.New("expected a JSON integer")
		}
	case FieldObject:
		var object map[string]json.RawMessage
		if err := strictjson.DecodeBytes(trimmed, &object); err != nil || object == nil {
			return errors.New("expected a strict JSON object")
		}
	default:
		return errors.New("undeclared field kind")
	}
	return nil
}
