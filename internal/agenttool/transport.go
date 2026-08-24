package agenttool

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

// Builtin is one in-process tool implementation: strict JSON in,
// strict JSON out.
type Builtin func(context.Context, json.RawMessage) (json.RawMessage, error)

// Executor invokes manuals over their declared native transports.
type Executor struct {
	builtins map[string]Builtin
	client   *http.Client
}

// NewExecutor returns the serving executor: no builtins registered,
// redirects refused, and http endpoints that resolve to loopback,
// private, or link-local addresses refused at dial time. Every
// network-facing surface uses this constructor.
func NewExecutor() *Executor {
	return &Executor{builtins: map[string]Builtin{}, client: newTransportClient(false)}
}

// NewOperatorExecutor returns the operator's executor: identical
// policy except that loopback and private endpoints are reachable,
// because an operator invoking local tooling from the CLI is not a
// server fetching on a client's behalf. Redirects stay refused.
func NewOperatorExecutor() *Executor {
	return &Executor{builtins: map[string]Builtin{}, client: newTransportClient(true)}
}

// registerBuiltin binds one in-process implementation to a manual name.
func (e *Executor) registerBuiltin(name string, implementation Builtin) error {
	if !manualNamePattern.MatchString(name) || implementation == nil {
		return fmt.Errorf("agent tool: invalid builtin registration %q", name)
	}
	if _, exists := e.builtins[name]; exists {
		return fmt.Errorf("agent tool: builtin %q is already registered", name)
	}
	e.builtins[name] = implementation
	return nil
}

// Invoke validates arguments against the manual and executes it over
// its declared transport, returning strict JSON or a typed error.
func (e *Executor) Invoke(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	if ctx == nil {
		return nil, errors.New("agent tool: nil invoke context")
	}
	if err := validateArguments(manual, arguments); err != nil {
		return nil, err
	}
	switch manual.Transport.Kind {
	case TransportBuiltin:
		implementation, registered := e.builtins[manual.Name]
		if !registered {
			return nil, fmt.Errorf("agent tool: builtin %q is not registered", manual.Name)
		}
		result, err := implementation(ctx, arguments)
		if err != nil {
			return nil, fmt.Errorf("agent tool: %q failed: %w", manual.Name, err)
		}
		return boundedResult(manual.Name, result)
	case TransportHTTP:
		return e.invokeHTTP(ctx, manual, arguments)
	case TransportArgv:
		return invokeArgv(ctx, manual, arguments)
	}
	return nil, fmt.Errorf("agent tool: manual %q transport kind is not declared", manual.Name)
}

func (e *Executor) invokeHTTP(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, manual.Transport.URL, bytes.NewReader(arguments))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", ManualMediaType)
	response, err := e.client.Do(request)
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
	return boundedResult(manual.Name, body)
}

func invokeArgv(ctx context.Context, manual Manual, arguments json.RawMessage) (json.RawMessage, error) {
	command := exec.CommandContext(ctx, manual.Transport.Program, manual.Transport.Args...)
	command.Stdin = bytes.NewReader(arguments)
	var stdout bytes.Buffer
	command.Stdout, command.Stderr = &stdout, io.Discard
	if err := command.Run(); err != nil {
		return nil, fmt.Errorf("agent tool: %q exited: %w", manual.Name, err)
	}
	// Argv tools speak text; the result is the stdout text as one JSON
	// string so every transport returns strict JSON to the loop.
	encoded, err := json.Marshal(strings.TrimRight(stdout.String(), "\r\n"))
	if err != nil {
		return nil, err
	}
	return boundedResult(manual.Name, encoded)
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
