// Package agenttool owns the agent tool authority: UTCP-style manuals
// describing each callable tool -- its effect class, typed arguments,
// and native transport binding -- published to OvergoDB under registered
// aliases so orchestration resolves tools from the store, never from
// code alone. An unregistered tool is not callable.
package agenttool

import (
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strings"

	"overgo/internal/artifact"
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

// Effect classifies what a tool invocation does to the world.
type Effect string

const (
	// EffectInspection reads state and changes nothing.
	EffectInspection Effect = "inspection"
	// EffectMutation changes state and is gated behind inspection.
	EffectMutation Effect = "mutation"
)

// Valid reports whether the effect is a declared class.
func (effect Effect) Valid() bool {
	return effect == EffectInspection || effect == EffectMutation
}

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
	Required    bool      `json:"required,omitempty"`
	Description string    `json:"description,omitempty"`
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
)

// Valid reports whether the transport kind is a declared path.
func (kind TransportKind) Valid() bool {
	return kind == TransportBuiltin || kind == TransportHTTP || kind == TransportArgv
}

// Transport binds a manual to its native invocation path.
type Transport struct {
	Kind TransportKind `json:"kind"`
	// URL is the strict-JSON POST endpoint for the http transport.
	URL string `json:"url,omitempty"`
	// Program and Args are the fixed executable and leading argument
	// words for the argv transport; the call payload rides on stdin.
	Program string   `json:"program,omitempty"`
	Args    []string `json:"args,omitempty"`
}

// Manual is one durable tool description: what the tool is, what it
// does to the world, what it takes, and how it is natively invoked.
type Manual struct {
	ID          artifact.ID `json:"-"`
	Version     uint16      `json:"version"`
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Effect      Effect      `json:"effect"`
	Arguments   []Field     `json:"arguments,omitempty"`
	Transport   Transport   `json:"transport"`
}

var manualNamePattern = regexp.MustCompile(`^[a-z][a-z0-9_.-]*$`)

var manualCodec = artifact.JSONDocumentCodec(
	"agent tool manual", artifact.KindRecipe, ManualMediaType, ManualSchema,
	func(manual *Manual) error { return manual.validate() },
	func(manual Manual) artifact.ID { return manual.ID },
	func(manual *Manual, id artifact.ID) { manual.ID = id },
	func(manual Manual) Manual {
		manual.Arguments = slices.Clone(manual.Arguments)
		manual.Transport.Args = slices.Clone(manual.Transport.Args)
		return manual
	},
)

// NewManual canonicalizes and identifies one manual.
func NewManual(manual Manual) (Manual, error) {
	manual.Version = ManualVersion
	return manualCodec.New(manual)
}

// Content returns the canonical committed bytes of the manual.
func (manual Manual) Content() ([]byte, error) {
	return manualCodec.ContentBytes(manual)
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
	return manual.Transport.validate(manual.Name)
}

func (transport Transport) validate(name string) error {
	switch transport.Kind {
	case TransportBuiltin:
		if transport.URL != "" || transport.Program != "" || len(transport.Args) != 0 {
			return fmt.Errorf("agent tool: builtin manual %q must not bind an endpoint or program", name)
		}
	case TransportHTTP:
		parsed, err := url.Parse(transport.URL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
			return fmt.Errorf("agent tool: http manual %q requires an absolute http(s) endpoint", name)
		}
		if transport.Program != "" || len(transport.Args) != 0 {
			return fmt.Errorf("agent tool: http manual %q must not bind a program", name)
		}
	case TransportArgv:
		if transport.Program == "" || transport.URL != "" {
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
	default:
		return fmt.Errorf("agent tool: manual %q transport kind is not declared", name)
	}
	return nil
}
