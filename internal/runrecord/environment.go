package runrecord

import (
	"errors"
	"strings"

	"overgo/internal/artifact"
)

const (
	EnvironmentVersion   uint16 = 1
	EnvironmentMediaType        = "application/vnd.overgo.run-environment+json"
	EnvironmentSchema           = "overgo/run-environment/v1"
	maxEnvironmentBytes         = 512
)

var environmentCodec = evidenceDocumentCodec(
	"run record environment", EnvironmentMediaType, EnvironmentSchema, canonicalizeEnvironment,
	func(value Environment) artifact.ID { return value.ID },
	func(value *Environment, id artifact.ID) { value.ID = id }, nil,
)

// Environment: immutable execution platform identity.
type Environment struct {
	Version uint16      `json:"version"`
	Host    string      `json:"host"`
	OS      string      `json:"os"`
	Arch    string      `json:"arch"`
	Device  string      `json:"device"`
	Backend string      `json:"backend"`
	Driver  string      `json:"driver"`
	Runtime string      `json:"runtime,omitempty"`
	ID      artifact.ID `json:"-"`
}

// NewEnvironment identifies a typed execution-platform record.
func NewEnvironment(environment Environment) (Environment, error) {
	environment.Version = EnvironmentVersion
	environment.ID = artifact.ID{}
	return environmentCodec.New(environment)
}

func (e Environment) ValidateIdentity() error {
	return environmentCodec.ValidateIdentity(e)
}

func (e Environment) Content() (artifact.Content, error) {
	return environmentCodec.Content(e)
}

func (e Environment) Batch(key string) (artifact.Batch, error) {
	content, err := e.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	return artifact.NewDocumentBatch(key, []artifact.Content{content}, nil, nil)
}

func canonicalizeEnvironment(environment *Environment) error {
	if environment == nil || environment.Version != EnvironmentVersion ||
		!validEnvironmentField(environment.Host) || !validEnvironmentField(environment.OS) ||
		!validEnvironmentField(environment.Arch) || !validEnvironmentField(environment.Device) ||
		!validEnvironmentField(environment.Backend) || !validEnvironmentField(environment.Driver) ||
		environment.Runtime != "" && !validEnvironmentField(environment.Runtime) {
		return errors.New("run record: invalid environment")
	}
	return nil
}

func validEnvironmentField(value string) bool {
	return value != "" && len(value) <= maxEnvironmentBytes && strings.TrimSpace(value) == value &&
		!strings.ContainsAny(value, "\x00\r\n")
}
