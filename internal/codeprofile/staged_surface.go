package codeprofile

import (
	"fmt"
	"os"

	"overgo/internal/jsonfile"
)

// StagedSurfaceEntry declares one exported symbol whose production
// consumer is deliberately deferred: the declaration names the reason
// and the trigger that retires it, so accepted-but-unconsumed surface
// is a reviewed, versioned decision instead of a ratchet violation.
type StagedSurfaceEntry struct {
	Package string `json:"package"`
	Name    string `json:"name"`
	Reason  string `json:"reason"`
	// ConsumerTrigger states what work is expected to consume the
	// symbol; a reviewer prunes entries whose trigger has landed.
	ConsumerTrigger string `json:"consumer_trigger"`
}

// StagedSurfaceDeclaration is the versioned staged-surface file.
type StagedSurfaceDeclaration struct {
	Version uint16               `json:"version"`
	Doc     string               `json:"doc,omitempty"`
	Staged  []StagedSurfaceEntry `json:"staged"`
}

// LoadStagedSurface reads the declaration; an absent file declares
// nothing and stages nothing. Decoding is strict because this file is
// gate authority: a duplicate name that silently last-wins would drop
// a reviewed entry, so it refuses like every other gate document.
func LoadStagedSurface(path string) (StagedSurfaceDeclaration, error) {
	var declaration StagedSurfaceDeclaration
	if _, err := os.Stat(path); err != nil {
		return StagedSurfaceDeclaration{Version: 1}, nil
	}
	if err := jsonfile.DecodeStrict(path, &declaration); err != nil {
		return StagedSurfaceDeclaration{}, err
	}
	if declaration.Version != 1 {
		return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: unsupported staged-surface version %d", declaration.Version)
	}
	for _, entry := range declaration.Staged {
		if entry.Package == "" || entry.Name == "" || entry.Reason == "" || entry.ConsumerTrigger == "" {
			return StagedSurfaceDeclaration{}, fmt.Errorf("codeprofile: staged entry %s.%s needs package, name, reason, and consumer trigger", entry.Package, entry.Name)
		}
	}
	return declaration, nil
}

// PartitionStagedSurface splits new unconsumed declarations into those
// the staged declaration accepts and those that remain blocking.
func PartitionStagedSurface(
	unconsumed []ConsumerDeclaration,
	declaration StagedSurfaceDeclaration,
) (accepted, blocking []ConsumerDeclaration) {
	staged := make(map[string]bool, len(declaration.Staged))
	for _, entry := range declaration.Staged {
		staged[entry.Package+"\x00"+entry.Name] = true
	}
	for _, candidate := range unconsumed {
		if staged[candidate.Package+"\x00"+candidate.Name] {
			accepted = append(accepted, candidate)
		} else {
			blocking = append(blocking, candidate)
		}
	}
	return accepted, blocking
}
