package apimanifest

import (
	"cmp"
	"encoding/json"
	"slices"
)

// ChangeClass classifies one manifest difference by its compatibility impact.
type ChangeClass string

const (
	// ChangeAdd marks an entry present only in the current manifest.
	ChangeAdd ChangeClass = "additive"
	// ChangeRemove marks an entry present only in the previous manifest.
	ChangeRemove ChangeClass = "removal"
	// ChangeContract marks an interface entry whose content changed.
	ChangeContract ChangeClass = "contract-change"
	// ChangeEvidence marks a changed evidence-only entry, such as an
	// authority or the release identity.
	ChangeEvidence ChangeClass = "evidence-only"
	// ChangeBuildSelection marks a changed build context.
	ChangeBuildSelection ChangeClass = "build-selection"
)

// Change records one classified difference between two manifests.
type Change struct {
	Class ChangeClass `json:"class"`
	Kind  string      `json:"kind"`
	Key   string      `json:"key"`
}

// Compare validates both manifests and returns their differences ordered by
// kind then key.
func Compare(previous, current Manifest) ([]Change, error) {
	if err := previous.Validate(); err != nil {
		return nil, err
	}
	if err := current.Validate(); err != nil {
		return nil, err
	}
	var changes []Change
	changes = append(changes, compareEntries("build", ChangeBuildSelection, previous.BuildContexts, current.BuildContexts, func(value BuildContext) string { return value.ID })...)
	changes = append(changes, compareEntries("binary", ChangeContract, previous.Binaries, current.Binaries, func(value Binary) string { return value.Name })...)
	changes = append(changes, compareEntries("route", ChangeContract, previous.Routes, current.Routes, func(value Route) string { return value.Method + " " + value.Path })...)
	changes = append(changes, compareEntries("document", ChangeContract, previous.Documents, current.Documents, func(value Document) string { return documentKey(value.Kind, value.MediaType, value.Schema) })...)
	changes = append(changes, compareEntries("module", ChangeContract, previous.Modules, current.Modules, func(value Module) string { return value.ID })...)
	changes = append(changes, compareEntries("protocol", ChangeContract, previous.Protocols, current.Protocols, func(value Protocol) string { return value.Name })...)
	changes = append(changes, compareEntries("workspace", ChangeContract, previous.Workspaces, current.Workspaces, func(value Workspace) string { return value.Name })...)
	changes = append(changes, compareEntries("authority", ChangeEvidence, previous.Authorities, current.Authorities, func(value Authority) string { return value.Name })...)
	if previous.Release != current.Release || previous.SourceIdentity != current.SourceIdentity {
		changes = append(changes, Change{Class: ChangeEvidence, Kind: "release", Key: current.Release})
	}
	slices.SortFunc(changes, func(left, right Change) int {
		return cmp.Or(cmp.Compare(left.Kind, right.Kind), cmp.Compare(left.Key, right.Key))
	})
	return changes, nil
}

func compareEntries[T any](kind string, changed ChangeClass, previous, current []T, key func(T) string) []Change {
	before := make(map[string][]byte, len(previous))
	for _, value := range previous {
		before[key(value)], _ = json.Marshal(value)
	}
	after := make(map[string][]byte, len(current))
	for _, value := range current {
		after[key(value)], _ = json.Marshal(value)
	}
	changes := make([]Change, 0)
	for entry, left := range before {
		right, found := after[entry]
		switch {
		case !found:
			changes = append(changes, Change{Class: ChangeRemove, Kind: kind, Key: entry})
		case !slices.Equal(left, right):
			changes = append(changes, Change{Class: changed, Kind: kind, Key: entry})
		}
	}
	for entry := range after {
		if _, found := before[entry]; !found {
			changes = append(changes, Change{Class: ChangeAdd, Kind: kind, Key: entry})
		}
	}
	return changes
}
