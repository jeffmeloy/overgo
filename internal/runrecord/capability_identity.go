package runrecord

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	// CapabilityIdentityMediaType is the canonical wire media type.
	CapabilityIdentityMediaType = "application/vnd.overgo.capability-identity+json"
	// CapabilityIdentitySchema is the canonical capability identity schema.
	CapabilityIdentitySchema = "overgo/capability-identity/v1"
)

// CapabilityTransportKind names a deterministic invocation transport.
type CapabilityTransportKind string

const (
	// CapabilityTransportBuiltin binds an in-process implementation.
	CapabilityTransportBuiltin CapabilityTransportKind = "builtin"
	// CapabilityTransportHTTP binds a strict request-response HTTP endpoint.
	CapabilityTransportHTTP CapabilityTransportKind = "http"
	// CapabilityTransportArgv binds a program and fixed argument prefix.
	CapabilityTransportArgv CapabilityTransportKind = "argv"
	// CapabilityTransportMCPHTTP binds an MCP-over-HTTP target.
	CapabilityTransportMCPHTTP CapabilityTransportKind = "mcp-http"
	// CapabilityTransportHTTPJSONStream binds a streaming HTTP endpoint.
	CapabilityTransportHTTPJSONStream CapabilityTransportKind = "http-json-stream"
	// CapabilityTransportPeer binds an enrolled peer endpoint.
	CapabilityTransportPeer CapabilityTransportKind = "peer"
)

// CapabilityTransport is the complete runnable transport authority.
type CapabilityTransport struct {
	Kind     CapabilityTransportKind `json:"kind"`
	Endpoint string                  `json:"endpoint,omitempty"`
	Protocol string                  `json:"protocol"`
	Target   string                  `json:"target,omitempty"`
	Args     []string                `json:"args,omitempty"`
}

// CapabilityPlatform identifies the required operating platform.
type CapabilityPlatform struct {
	OS          string `json:"os"`
	Arch        string `json:"arch"`
	Accelerator string `json:"accelerator,omitempty"`
}

// CapabilityResourceEnvelope bounds one activation's resource use.
type CapabilityResourceEnvelope struct {
	MaxInputBytes  uint64 `json:"max_input_bytes"`
	MaxOutputBytes uint64 `json:"max_output_bytes"`
	MaxConcurrent  uint32 `json:"max_concurrent"`
	CPUThreads     uint32 `json:"cpu_threads"`
	HostBytes      uint64 `json:"host_bytes"`
	DeviceBytes    uint64 `json:"device_bytes,omitempty"`
}

// CapabilityIdentity binds a runnable implementation to every fact that can
// change compatibility or safe activation.
type CapabilityIdentity struct {
	Version        uint16                     `json:"version"`
	Implementation artifact.ID                `json:"implementation"`
	Release        string                     `json:"release"`
	Transport      CapabilityTransport        `json:"transport"`
	Schema         artifact.ID                `json:"schema"`
	Platform       CapabilityPlatform         `json:"platform"`
	Resources      CapabilityResourceEnvelope `json:"resources"`
	ID             artifact.ID                `json:"-"`
}

var capabilityIdentityCodec = artifact.JSONDocumentCodec(
	"capability identity", artifact.KindProfile, CapabilityIdentityMediaType, CapabilityIdentitySchema,
	canonicalizeCapabilityIdentity,
	func(value CapabilityIdentity) artifact.ID { return value.ID },
	func(value *CapabilityIdentity, id artifact.ID) { value.ID = id },
	func(value CapabilityIdentity) CapabilityIdentity {
		value.Transport.Args = slices.Clone(value.Transport.Args)
		return value
	},
)

// Identify validates and identifies one exact capability.
func (value CapabilityIdentity) Identify() (CapabilityIdentity, error) {
	if value.Version != 0 && value.Version != artifact.InitialDocumentVersion {
		return CapabilityIdentity{}, errors.New("run record: unsupported capability identity version")
	}
	return capabilityIdentityCodec.NewInitial(value)
}

// RequireCapabilityIdentity resolves one exact capability identity.
func RequireCapabilityIdentity(ctx context.Context, reader artifact.Reader, id artifact.ID) (CapabilityIdentity, error) {
	return capabilityIdentityCodec.Require(ctx, reader, id)
}

// ValidateIdentity verifies that the value matches its immutable identity.
func (value CapabilityIdentity) ValidateIdentity() error {
	return capabilityIdentityCodec.ValidateIdentity(value)
}

// Content returns the canonical capability document content.
func (value CapabilityIdentity) Content() (artifact.Content, error) {
	return capabilityIdentityCodec.Content(value)
}

// Lineage links a capability to its implementation and schema.
func (value CapabilityIdentity) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Implementation, value.Schema)
}

func canonicalizeCapabilityIdentity(value *CapabilityIdentity) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Implementation.Valid() ||
		value.Schema.Kind() != artifact.KindProfile || !boundedCapabilityText(value.Release) ||
		!boundedCapabilityText(value.Transport.Protocol) || !boundedCapabilityText(value.Platform.OS) ||
		!boundedCapabilityText(value.Platform.Arch) || value.Resources.MaxInputBytes == 0 ||
		value.Resources.MaxOutputBytes == 0 || value.Resources.MaxConcurrent == 0 ||
		value.Resources.CPUThreads == 0 || value.Resources.HostBytes == 0 {
		return errors.New("run record: incomplete capability identity")
	}
	switch value.Transport.Kind {
	case CapabilityTransportBuiltin:
		if value.Transport.Endpoint != "" || value.Transport.Target != "" || len(value.Transport.Args) != 0 {
			return errors.New("run record: builtin capability has an endpoint")
		}
	case CapabilityTransportHTTP, CapabilityTransportHTTPJSONStream, CapabilityTransportPeer:
		if !boundedCapabilityText(value.Transport.Endpoint) {
			return errors.New("run record: transported capability requires an endpoint")
		}
		if value.Transport.Target != "" || len(value.Transport.Args) != 0 {
			return errors.New("run record: capability transport carries foreign binding fields")
		}
	case CapabilityTransportArgv:
		if !boundedCapabilityText(value.Transport.Endpoint) || value.Transport.Target != "" ||
			slices.ContainsFunc(value.Transport.Args, func(argument string) bool { return !boundedCapabilityText(argument) }) {
			return errors.New("run record: invalid argv capability binding")
		}
	case CapabilityTransportMCPHTTP:
		if !boundedCapabilityText(value.Transport.Endpoint) || !boundedCapabilityText(value.Transport.Target) || len(value.Transport.Args) != 0 {
			return errors.New("run record: invalid MCP capability binding")
		}
	default:
		return errors.New("run record: foreign capability transport")
	}
	if value.Resources.DeviceBytes == 0 != (value.Platform.Accelerator == "") ||
		value.Platform.Accelerator != "" && !boundedCapabilityText(value.Platform.Accelerator) {
		return errors.New("run record: accelerator and device envelope differ")
	}
	return nil
}

func boundedCapabilityText(value string) bool {
	return value != "" && strings.TrimSpace(value) == value && textcheck.Bounded(value, len(value), "\x00\r\n\t")
}
