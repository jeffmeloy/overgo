// Package agenttool owns the agent tool authority: UTCP-style manuals
// describing each callable tool -- its effect class, typed arguments,
// and native transport binding -- published to OvergoDB under registered
// aliases so orchestration resolves tools from the store, never from
// code alone. An unregistered tool is not callable.
package agenttool

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/invocation"
	"overgo/internal/runrecord"
	"overgo/internal/textcheck"
)

const (
	// ManualVersion is the manual document revision this package writes.
	ManualVersion uint16 = 1
	// ManualMediaType is the manual document wire label.
	ManualMediaType = "application/vnd.overgo.agent-tool-manual+json"
	// ManualSchema is the manual document wire schema.
	ManualSchema = "overgo/agent-tool-manual/v1"
)

// Effect is the transport-neutral invocation class.
type Effect = invocation.Class

const (
	// EffectInspection reads state and changes nothing.
	EffectInspection = invocation.ClassInspection
	// EffectMutation changes state and is gated behind inspection.
	EffectMutation = invocation.ClassMutation
)

// FieldKind types one manual argument.
type FieldKind string

const (
	// FieldString accepts a JSON string.
	FieldString FieldKind = "string"
	// FieldInteger accepts a JSON integer.
	FieldInteger FieldKind = "integer"
	// FieldNumber accepts a JSON number.
	FieldNumber FieldKind = "number"
	// FieldBoolean accepts a JSON boolean.
	FieldBoolean FieldKind = "boolean"
	// FieldObject accepts a strict JSON object.
	FieldObject FieldKind = "object"
)

// Valid reports whether the field kind is a declared type.
func (kind FieldKind) Valid() bool {
	switch kind {
	case FieldString, FieldInteger, FieldNumber, FieldBoolean, FieldObject:
		return true
	}
	return false
}

// Field declares one named, typed manual argument.
type Field struct {
	Name        string    `json:"name"`
	Kind        FieldKind `json:"kind"`
	Required    bool      `json:"required,omitzero"`
	Description string    `json:"description,omitzero"`
}

// EffectScope is the transport-neutral invocation target scope.
type EffectScope = invocation.Scope

const (
	// EffectScopeWorkspace targets files inside the leased worktree.
	EffectScopeWorkspace = invocation.ScopeWorkspace
	// EffectScopeRepository targets the common artifact repository.
	EffectScopeRepository = invocation.ScopeRepository
	// EffectScopeHost targets host state outside workspace and repository.
	EffectScopeHost = invocation.ScopeHost
	// EffectScopeExternal targets systems beyond the host boundary.
	EffectScopeExternal = invocation.ScopeExternal
)

// EffectTargetBinding declares how a concrete target is resolved. Argument
// names are explicit authority; no consumer guesses path semantics from names.
type EffectTargetBinding struct {
	Scope    EffectScope `json:"scope"`
	Argument string      `json:"argument,omitzero"`
	Value    string      `json:"value,omitzero"`
}

// EffectCeiling is the manual's static upper bound on invocation effects.
// Concrete targets are resolved only after strict argument validation.
type EffectCeiling struct {
	Targets      []EffectTargetBinding `json:"targets,omitempty"`
	Destructive  bool                  `json:"destructive,omitzero"`
	Privileged   bool                  `json:"privileged,omitzero"`
	Irreversible bool                  `json:"irreversible,omitzero"`
}

// TransportKind names a native invocation path.
type TransportKind string

const (
	// TransportBuiltin invokes a Go function registered in-process.
	TransportBuiltin TransportKind = "builtin"
	// TransportHTTP posts strict JSON to a bounded HTTP endpoint.
	TransportHTTP TransportKind = "http"
	// TransportArgv executes a fixed program with argument words, no shell.
	TransportArgv TransportKind = "argv"
	// TransportMCPHTTP invokes one exact MCP tool over bounded HTTP JSON-RPC.
	TransportMCPHTTP TransportKind = "mcp-http"
	// TransportHTTPJSONStream receives sequenced JSON values over one HTTP response.
	TransportHTTPJSONStream TransportKind = "http-json-stream"
)

// Valid reports whether the transport kind is a declared path.
func (kind TransportKind) Valid() bool {
	return kind == TransportBuiltin || kind == TransportHTTP || kind == TransportArgv ||
		kind == TransportMCPHTTP || kind == TransportHTTPJSONStream
}

// Transport binds a manual to its native invocation path.
type Transport struct {
	Kind TransportKind `json:"kind"`
	// URL is the strict-JSON POST endpoint for the http transport.
	URL string `json:"url,omitzero"`
	// Program and Args are the fixed executable and leading argument
	// words for the argv transport; the call payload rides on stdin.
	Program string   `json:"program,omitzero"`
	Args    []string `json:"args,omitempty"`
	// Target binds an MCP manual to one remote tool. Protocol identifies the
	// adapter contract when the manual is capability-bound. Both are authority.
	Target   string `json:"target,omitzero"`
	Protocol string `json:"protocol,omitzero"`
}

// Manual is one durable tool description: what the tool is, what it
// does to the world, what it takes, and how it is natively invoked.
type Manual struct {
	ID          artifact.ID   `json:"-"`
	Version     uint16        `json:"version"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Effect      Effect        `json:"effect"`
	Ceiling     EffectCeiling `json:"ceiling,omitempty"`
	Arguments   []Field       `json:"arguments,omitempty"`
	Transport   Transport     `json:"transport"`
	Capability  artifact.ID   `json:"capability,omitzero"`
	// CapabilityIdentity supplies new canonical content during catalog
	// publication; only Capability is stored in the manual itself.
	CapabilityIdentity *runrecord.CapabilityIdentity `json:"-"`
}

var manualNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_.-]*$`)

var manualCodec = artifact.JSONDocumentCodec(
	"agent tool manual", artifact.KindRecipe, ManualMediaType, ManualSchema,
	func(manual *Manual) error { return manual.validate() },
	func(manual Manual) artifact.ID { return manual.ID },
	func(manual *Manual, id artifact.ID) { manual.ID = id },
	func(manual Manual) Manual {
		manual.Arguments = slices.Clone(manual.Arguments)
		manual.Ceiling.Targets = slices.Clone(manual.Ceiling.Targets)
		manual.Transport.Args = slices.Clone(manual.Transport.Args)
		if manual.CapabilityIdentity != nil {
			identity := *manual.CapabilityIdentity
			manual.CapabilityIdentity = &identity
		}
		return manual
	},
)

// NewManual canonicalizes and identifies one manual.
func NewManual(manual Manual) (Manual, error) {
	manual.Version = ManualVersion
	if manual.CapabilityIdentity != nil {
		identity, err := manual.CapabilityIdentity.Identify()
		if err != nil {
			return Manual{}, err
		}
		if manual.Capability.Valid() && manual.Capability != identity.ID {
			return Manual{}, errors.New("agent tool: supplied capability identity differs")
		}
		manual.Capability, manual.CapabilityIdentity = identity.ID, &identity
	}
	return manualCodec.New(manual)
}

// Content returns the canonical committed bytes of the manual.
func (manual Manual) Content() ([]byte, error) {
	return manualCodec.ContentBytes(manual)
}

func appendManualDocuments(
	contents *[]artifact.Content,
	lineage *[]artifact.Lineage,
	manual Manual,
	capabilities map[artifact.ID]bool,
) error {
	if manual.CapabilityIdentity != nil && !capabilities[manual.Capability] {
		content, err := manual.CapabilityIdentity.Content()
		if err != nil {
			return err
		}
		*contents = append(*contents, content)
		*lineage = append(*lineage, manual.CapabilityIdentity.Lineage()...)
		capabilities[manual.Capability] = true
	}
	content, err := manualCodec.Content(manual)
	if err != nil {
		return err
	}
	*contents = append(*contents, content)
	if manual.Capability.Valid() {
		*lineage = append(*lineage, artifact.DependencyLineage(manual.ID, manual.Capability)...)
	}
	return nil
}

// RequireManual loads one exact manual by immutable identity.
func RequireManual(ctx context.Context, reader artifact.Reader, id artifact.ID) (Manual, error) {
	manual, err := manualCodec.Require(ctx, reader, id)
	if err != nil || !manual.Capability.Valid() {
		return manual, err
	}
	identity, err := runrecord.RequireCapabilityIdentity(ctx, reader, manual.Capability)
	if err != nil || identity.ID != manual.Capability || !manualTransportMatchesCapability(manual.Transport, identity.Transport) {
		return Manual{}, errors.Join(errors.New("agent tool: manual capability cannot be resolved exactly"), err)
	}
	return manual, nil
}

func (manual *Manual) validate() error {
	if manual.Version != ManualVersion {
		return errors.New("agent tool: unsupported manual version")
	}
	if !manualNamePattern.MatchString(manual.Name) || !textcheck.Bounded(manual.Name, len(manual.Name), "\x00\r\n") {
		return fmt.Errorf("agent tool: invalid manual name %q", manual.Name)
	}
	if manual.Description == "" || !textcheck.Bounded(manual.Description, len(manual.Description), "\x00\r\n") {
		return errors.New("agent tool: manual description is required and bounded")
	}
	if !manual.Effect.Valid() {
		return fmt.Errorf("agent tool: manual %q effect must declare inspection or mutation", manual.Name)
	}
	if manual.Capability.Valid() {
		if manual.Capability.Kind() != artifact.KindProfile || manual.CapabilityIdentity != nil &&
			(manual.CapabilityIdentity.ID != manual.Capability || manual.CapabilityIdentity.ValidateIdentity() != nil ||
				!manualTransportMatchesCapability(manual.Transport, manual.CapabilityIdentity.Transport)) {
			return errors.New("agent tool: manual capability identity differs")
		}
	} else if manual.CapabilityIdentity != nil {
		return errors.New("agent tool: manual capability identity is absent")
	} else if manual.Transport.Kind != TransportMCPHTTP && manual.Transport.Protocol != "" {
		return errors.New("agent tool: unbound manual declares a capability protocol")
	}
	seen := map[string]bool{}
	for _, field := range manual.Arguments {
		if !manualNamePattern.MatchString(field.Name) || seen[field.Name] {
			return fmt.Errorf("agent tool: manual %q argument names must be unique and lowercase", manual.Name)
		}
		if !field.Kind.Valid() {
			return fmt.Errorf("agent tool: manual %q argument %q has no declared kind", manual.Name, field.Name)
		}
		if field.Description != "" && !textcheck.Bounded(field.Description, len(field.Description), "\x00\r\n") {
			return fmt.Errorf("agent tool: manual %q argument %q description exceeds the bound", manual.Name, field.Name)
		}
		seen[field.Name] = true
	}
	for _, binding := range manual.Ceiling.Targets {
		if !binding.Scope.Valid() || (binding.Argument == "") == (binding.Value == "") {
			return fmt.Errorf("agent tool: manual %q effect target must declare one source and a valid scope", manual.Name)
		}
		if binding.Argument != "" {
			field, found := func() (Field, bool) {
				for _, candidate := range manual.Arguments {
					if candidate.Name == binding.Argument {
						return candidate, true
					}
				}
				return Field{}, false
			}()
			if !found || field.Kind != FieldString {
				return fmt.Errorf("agent tool: manual %q effect target argument %q must be a declared string", manual.Name, binding.Argument)
			}
		}
	}
	return manual.Transport.validate(manual.Name)
}

func manualTransportMatchesCapability(transport Transport, capability runrecord.CapabilityTransport) bool {
	if string(transport.Kind) != string(capability.Kind) || transport.Protocol != capability.Protocol {
		return false
	}
	switch transport.Kind {
	case TransportBuiltin:
		return capability.Endpoint == "" && capability.Target == "" && len(capability.Args) == 0
	case TransportHTTP, TransportHTTPJSONStream:
		return transport.URL == capability.Endpoint && capability.Target == "" && len(capability.Args) == 0
	case TransportArgv:
		return transport.Program == capability.Endpoint && slices.Equal(transport.Args, capability.Args) && capability.Target == ""
	case TransportMCPHTTP:
		return transport.URL == capability.Endpoint && transport.Target == capability.Target &&
			len(capability.Args) == 0
	default:
		return false
	}
}

func (transport Transport) validate(name string) error {
	switch transport.Kind {
	case TransportBuiltin:
		if transport.URL != "" || transport.Program != "" || len(transport.Args) != 0 || transport.Target != "" ||
			transport.Protocol != "" && !textcheck.Bounded(transport.Protocol, len(transport.Protocol), "\x00\r\n") {
			return fmt.Errorf("agent tool: builtin manual %q must not bind an endpoint or program", name)
		}
	case TransportHTTP, TransportHTTPJSONStream:
		parsed, err := url.Parse(transport.URL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
			return fmt.Errorf("agent tool: http manual %q requires an absolute http(s) endpoint", name)
		}
		if transport.Program != "" || len(transport.Args) != 0 || transport.Target != "" ||
			transport.Protocol != "" && !textcheck.Bounded(transport.Protocol, len(transport.Protocol), "\x00\r\n") {
			return fmt.Errorf("agent tool: http manual %q must not bind a program", name)
		}
	case TransportArgv:
		if transport.Program == "" || transport.URL != "" || transport.Target != "" ||
			transport.Protocol != "" && !textcheck.Bounded(transport.Protocol, len(transport.Protocol), "\x00\r\n") {
			return fmt.Errorf("agent tool: argv manual %q requires a program and no endpoint", name)
		}
		// A program is a bare command word resolved on PATH: a path
		// separator would let a manual point execution at arbitrary
		// files, and the operator's allowlist could not reason about it.
		if strings.ContainsAny(transport.Program, `/\`) || !textcheck.Bounded(transport.Program, len(transport.Program), "\x00\r\n") {
			return fmt.Errorf("agent tool: argv manual %q program must be a bare command word", name)
		}
		for _, word := range transport.Args {
			if !textcheck.Bounded(word, len(word), "\x00\r\n") {
				return fmt.Errorf("agent tool: argv manual %q argument word exceeds the bound", name)
			}
		}
	case TransportMCPHTTP:
		parsed, err := url.Parse(transport.URL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
			return fmt.Errorf("agent tool: mcp manual %q requires an absolute http(s) endpoint", name)
		}
		if transport.Program != "" || len(transport.Args) != 0 ||
			!manualNamePattern.MatchString(transport.Target) ||
			!textcheck.Bounded(transport.Protocol, len(transport.Protocol), "\x00\r\n") || transport.Protocol == "" {
			return fmt.Errorf("agent tool: mcp manual %q requires an exact target and protocol", name)
		}
	default:
		return fmt.Errorf("agent tool: manual %q transport kind is not declared", name)
	}
	return nil
}
