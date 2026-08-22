package artifact

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/pathidentity"
)

const maxLocationBytes = 32 << 10

// LocationKind defines physical artifact address class.
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

// LocationAction defines append-only availability transition.
type LocationAction uint8

const (
	_ LocationAction = iota
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

// Location defines current physical artifact address.
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

// LocationEvent defines availability set mutation.
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

// CanonicalLocalLocation validates a live local file or directory and records
// its absolute link-resolved spelling. Construction is intentionally separate
// from Location.Validate: historical removal events must remain readable after
// their files disappear.
func CanonicalLocalLocation(id ID, kind LocationKind, path string) (Location, error) {
	if !id.Valid() || kind != LocationFile && kind != LocationDirectory {
		return Location{}, errors.New("artifact: invalid canonical local location request")
	}
	canonical, err := pathidentity.Canonical(path)
	if err != nil {
		return Location{}, fmt.Errorf("artifact: canonical local location: %w", err)
	}
	info, err := os.Stat(canonical)
	if err != nil {
		return Location{}, fmt.Errorf("artifact: inspect canonical local location: %w", err)
	}
	if info.IsDir() != (kind == LocationDirectory) {
		return Location{}, errors.New("artifact: canonical local location kind differs")
	}
	location := Location{Artifact: id, Kind: kind, Value: canonical}
	if err := location.Validate(); err != nil {
		return Location{}, err
	}
	return location, nil
}

// SameLocalLocation compares two local addresses by filesystem identity. The
// artifact and location kind remain part of the identity; aliases cannot make
// one artifact's bytes satisfy another artifact's location claim.
func SameLocalLocation(left, right Location) (bool, error) {
	if err := left.Validate(); err != nil {
		return false, err
	}
	if err := right.Validate(); err != nil {
		return false, err
	}
	if left.Artifact != right.Artifact || left.Kind != right.Kind ||
		left.Kind != LocationFile && left.Kind != LocationDirectory {
		return false, nil
	}
	return pathidentity.Same(left.Value, right.Value)
}

// AvailablePath returns the first recorded live file or directory.
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
		canonical, canonicalErr := CanonicalLocalLocation(id, kind, path)
		if canonicalErr == nil {
			return canonical.Value, nil
		}
	}
	return "", fmt.Errorf("artifact: %s has no available %s", id, kind)
}
