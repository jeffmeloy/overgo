package runrecord

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	EnvironmentVersion   uint16 = 1
	EnvironmentMediaType        = "application/vnd.overgo.run-environment+json"
	EnvironmentSchema           = "overgo/run-environment/v1"
	maxEnvironmentBytes         = 512
)

var environmentContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: EnvironmentMediaType, Schema: EnvironmentSchema,
}

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

func NewEnvironment(host, osName, arch, device, backend, driver, runtime string) (Environment, error) {
	environment := Environment{
		Version: EnvironmentVersion, Host: host, OS: osName, Arch: arch,
		Device: device, Backend: backend, Driver: driver, Runtime: runtime,
	}
	if err := canonicalizeEnvironment(&environment); err != nil {
		return Environment{}, err
	}
	content, err := environmentContent(environment)
	if err != nil {
		return Environment{}, err
	}
	environment.ID, err = environmentContract.Identify(content)
	return environment, err
}

func ParseEnvironment(content []byte) (Environment, error) {
	var body Environment
	if err := strictjson.DecodeBytes(content, &body); err != nil {
		return Environment{}, fmt.Errorf("run record: decode environment: %w", err)
	}
	environment, err := NewEnvironment(
		body.Host, body.OS, body.Arch, body.Device, body.Backend, body.Driver, body.Runtime,
	)
	if err != nil {
		return Environment{}, err
	}
	canonical, err := environment.ContentBytes()
	if err != nil {
		return Environment{}, err
	}
	if !bytes.Equal(canonical, content) {
		return Environment{}, errors.New("run record: non-canonical environment content")
	}
	return environment, nil
}

func (e Environment) ValidateIdentity() error {
	if e.ID.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid environment identity")
	}
	canonical := e
	canonical.ID = artifact.ID{}
	if err := canonicalizeEnvironment(&canonical); err != nil {
		return err
	}
	canonical.ID = e.ID
	if e != canonical {
		return errors.New("run record: environment is not canonical")
	}
	canonical.ID = artifact.ID{}
	content, err := environmentContent(canonical)
	if err != nil {
		return err
	}
	if err := environmentContract.ValidateIdentity(e.ID, content); err != nil {
		return errors.New("run record: environment identity mismatch")
	}
	return nil
}

func (e Environment) ContentBytes() ([]byte, error) {
	if err := e.ValidateIdentity(); err != nil {
		return nil, err
	}
	e.ID = artifact.ID{}
	return environmentContent(e)
}

func (e Environment) Content() (artifact.Content, error) {
	id := e.ID
	content, err := e.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return environmentContract.Content(id, content)
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

func environmentContent(environment Environment) ([]byte, error) {
	environment.ID = artifact.ID{}
	content, err := json.Marshal(environment)
	if err != nil {
		return nil, fmt.Errorf("run record: encode environment: %w", err)
	}
	return content, nil
}
