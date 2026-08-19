package runrecord

import (
	"errors"
	"os"
	"runtime"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	EnvironmentVersion   uint16 = 1
	EnvironmentMediaType        = "application/vnd.overgo.run-environment+json"
	EnvironmentSchema           = "overgo/run-environment/v1"
	maxEnvironmentBytes         = 512
)

var environmentCodec = artifact.JSONDocumentCodec(
	"run record environment", artifact.KindEvidence, EnvironmentMediaType, EnvironmentSchema, canonicalizeEnvironment,
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

// CurrentEnvironment: process runtime identity.
func CurrentEnvironment(device, backend string) (Environment, error) {
	host, err := os.Hostname()
	if err != nil {
		host = "unknown"
	}
	return NewEnvironment(Environment{
		Host: host, OS: runtime.GOOS, Arch: runtime.GOARCH,
		Device: device, Backend: backend, Driver: "process", Runtime: runtime.Version(),
	})
}

func (e Environment) ValidateIdentity() error {
	return environmentCodec.ValidateIdentity(e)
}

func (e Environment) Content() (artifact.Content, error) {
	return environmentCodec.Content(e)
}

func (e Environment) Batch(key string) (artifact.Batch, error) {
	return environmentCodec.Batch(key, e, nil, nil)
}

func canonicalizeEnvironment(environment *Environment) error {
	valid := func(value string) bool { return textcheck.Bounded(value, maxEnvironmentBytes, "\x00\r\n") }
	if environment == nil || environment.Version != EnvironmentVersion ||
		!valid(environment.Host) || !valid(environment.OS) || !valid(environment.Arch) || !valid(environment.Device) ||
		!valid(environment.Backend) || !valid(environment.Driver) || environment.Runtime != "" && !valid(environment.Runtime) {
		return errors.New("run record: invalid environment")
	}
	return nil
}
