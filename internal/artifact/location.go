package artifact

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const maxLocationBytes = 32 << 10

// LocationKind: physical artifact address class
type LocationKind uint8

const (
	LocationInvalid LocationKind = iota
	LocationFile
	LocationDirectory
	LocationRemote
)

var locationKindNames = [...]string{
	LocationInvalid:   "invalid",
	LocationFile:      "file",
	LocationDirectory: "directory",
	LocationRemote:    "remote",
}

func (k LocationKind) String() string {
	if int(k) >= len(locationKindNames) {
		return locationKindNames[LocationInvalid]
	}
	return locationKindNames[k]
}

func (k LocationKind) MarshalJSON() ([]byte, error) {
	if k == LocationInvalid || int(k) >= len(locationKindNames) {
		return nil, errors.New("artifact: invalid location kind")
	}
	return json.Marshal(k.String())
}

func (k *LocationKind) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("artifact: decode location kind: %w", err)
	}
	for kind, name := range locationKindNames {
		if kind > 0 && value == name {
			*k = LocationKind(kind)
			return nil
		}
	}
	return fmt.Errorf("artifact: unknown location kind %q", value)
}

// LocationAction: append-only availability transition
type LocationAction uint8

const (
	LocationActionInvalid LocationAction = iota
	LocationAdd
	LocationRemove
)

func (a LocationAction) MarshalJSON() ([]byte, error) {
	switch a {
	case LocationAdd:
		return json.Marshal("add")
	case LocationRemove:
		return json.Marshal("remove")
	default:
		return nil, errors.New("artifact: invalid location action")
	}
}

func (a *LocationAction) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return fmt.Errorf("artifact: decode location action: %w", err)
	}
	switch value {
	case "add":
		*a = LocationAdd
	case "remove":
		*a = LocationRemove
	default:
		return fmt.Errorf("artifact: unknown location action %q", value)
	}
	return nil
}

// Location: current physical artifact address
type Location struct {
	Artifact ID           `json:"artifact"`
	Kind     LocationKind `json:"kind"`
	Value    string       `json:"value"`
}

func (l Location) Validate() error {
	if !l.Artifact.Valid() || l.Kind == LocationInvalid || int(l.Kind) >= len(locationKindNames) {
		return errors.New("artifact: invalid location identity or kind")
	}
	if l.Value == "" || len(l.Value) > maxLocationBytes || strings.TrimSpace(l.Value) != l.Value || strings.ContainsAny(l.Value, "\r\n") {
		return errors.New("artifact: invalid location value")
	}
	return nil
}

// LocationEvent: availability set mutation
type LocationEvent struct {
	Location
	Action LocationAction `json:"action"`
}

func (e LocationEvent) Validate() error {
	if err := e.Location.Validate(); err != nil {
		return err
	}
	if e.Action != LocationAdd && e.Action != LocationRemove {
		return errors.New("artifact: invalid location action")
	}
	return nil
}
