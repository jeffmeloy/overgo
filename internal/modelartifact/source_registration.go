package modelartifact

import (
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
)

// VerifySource binds every inventoried file to a previously verified source receipt.
// Source verification remains separate from architecture support and execution.
func (i Inventory) VerifySource(directory string, source hfrepo.SourceReceipt) error {
	root, err := filepath.Abs(directory)
	if err != nil {
		return err
	}
	descriptors := map[artifact.ID]artifact.Descriptor{}
	for _, descriptor := range i.Components {
		descriptors[descriptor.ID] = descriptor
	}
	seen := map[artifact.ID]bool{}
	for _, location := range i.Locations {
		if location.Kind != artifact.LocationFile {
			continue
		}
		relative, err := filepath.Rel(root, location.Value)
		descriptor, found := descriptors[location.Artifact]
		if err != nil || !found || !source.Covers(hfrepo.SourceFile{Path: filepath.ToSlash(relative), SHA256: location.Artifact.DigestHex(), Bytes: int64(descriptor.Size)}) {
			return fmt.Errorf("model artifact: inventoried payload %q is absent from the verified source", location.Value)
		}
		seen[location.Artifact] = true
	}
	for _, descriptor := range i.Components {
		if !seen[descriptor.ID] {
			return fmt.Errorf("model artifact: payload %s has no verified source location", descriptor.ID)
		}
	}
	return nil
}
