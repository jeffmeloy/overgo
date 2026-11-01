package modelartifact

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

// VerifySources proves each physical source still matches the frozen inventory.
// Missing descriptors or file locations are refused before publication.
// Verification neither copies source bytes nor establishes numerical acceptance.
func (i Inventory) VerifySources(ctx context.Context) error {
	if ctx == nil {
		return errors.New("model artifact: source verification requires a context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(i.Components) == 0 {
		return errors.New("model artifact: source inventory has no physical components")
	}
	descriptors := make(map[artifact.ID]artifact.Descriptor, len(i.Components))
	for _, descriptor := range i.Components {
		if err := descriptor.Validate(); err != nil {
			return err
		}
		if previous, ok := descriptors[descriptor.ID]; ok && previous != descriptor {
			return fmt.Errorf("model artifact: conflicting source descriptors for %s", descriptor.ID)
		}
		descriptors[descriptor.ID] = descriptor
	}
	for _, component := range i.Manifest.Components {
		if _, ok := descriptors[component.Artifact]; !ok {
			return fmt.Errorf("model artifact: source %q lacks its descriptor", component.Name)
		}
	}
	verified := map[artifact.ID]bool{}
	for _, location := range i.Locations {
		if location.Kind != artifact.LocationFile {
			continue
		}
		expected, ok := descriptors[location.Artifact]
		if !ok {
			return fmt.Errorf("model artifact: file location has no source descriptor for %s", location.Artifact)
		}
		actual, _, err := identifyFileContext(ctx, location.Value, expected.ID.Kind(), expected.MediaType)
		if err != nil {
			return fmt.Errorf("model artifact: verify source %s: %w", expected.ID, err)
		}
		if actual.ID != expected.ID || actual.Size != expected.Size {
			return fmt.Errorf("model artifact: source %s changed after inspection: %w", expected.ID, artifact.ErrContentMismatch)
		}
		verified[expected.ID] = true
	}
	for _, descriptor := range i.Components {
		if !verified[descriptor.ID] {
			return fmt.Errorf("model artifact: source %s has no verified file location", descriptor.ID)
		}
	}
	return ctx.Err()
}
