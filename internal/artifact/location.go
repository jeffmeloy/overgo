package artifact

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
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
	return marshalEnumJSON(int(k), locationKindNames[:], "location kind")
}

func (k *LocationKind) UnmarshalJSON(data []byte) error {
	return unmarshalEnumInto(k, data, locationKindNames[:], "location kind")
}

// LocationAction: append-only availability transition
type LocationAction uint8

const (
	LocationActionInvalid LocationAction = iota
	LocationAdd
	LocationRemove
)

var locationActionNames = [...]string{"invalid", "add", "remove"}

func (a LocationAction) MarshalJSON() ([]byte, error) {
	return marshalEnumJSON(int(a), locationActionNames[:], "location action")
}

func (a *LocationAction) UnmarshalJSON(data []byte) error {
	return unmarshalEnumInto(a, data, locationActionNames[:], "location action")
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

// AvailablePath: first recorded live file or directory.
func AvailablePath(ctx context.Context, reader Reader, id ID, kind LocationKind) (string, error) {
	if ctx == nil || reader == nil || !id.Valid() || kind != LocationFile && kind != LocationDirectory {
		return "", errors.New("artifact: invalid available-path query")
	}
	locations, err := reader.Locations(ctx, id)
	if err != nil {
		return "", err
	}
	for _, location := range locations {
		path := location.Value
		if kind == LocationDirectory && location.Kind == LocationFile {
			path = filepath.Dir(path)
		} else if location.Kind != kind {
			continue
		}
		info, statErr := os.Stat(path)
		if statErr == nil && info.IsDir() == (kind == LocationDirectory) {
			return path, nil
		}
	}
	return "", fmt.Errorf("artifact: %s has no available %s", id, kind)
}
