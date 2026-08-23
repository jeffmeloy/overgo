// Package artifact defines typed capability bundle storage.
package artifact

import (
	"context"
	"errors"
)

// ValidateCapabilityBundle verifies an instruction/resource manifest.
func ValidateCapabilityBundle(manifest Manifest) error {
	if manifest.ID.Kind() != KindProfile {
		return errors.New("artifact: capability bundle has invalid identity")
	}
	if err := manifest.Validate(); err != nil {
		return err
	}
	for _, component := range manifest.Components {
		if component.Role != ComponentInstruction && component.Role != ComponentResource {
			return errors.New("artifact: capability bundle has invalid component role")
		}
	}
	return nil
}

// LoadManifestComponent loads one named member without opening siblings.
func LoadManifestComponent(
	ctx context.Context,
	reader Reader,
	manifest Manifest,
	role ComponentRole,
	name string,
) (Content, error) {
	if ctx == nil || reader == nil || name == "" {
		return Content{}, errors.New("artifact: invalid manifest component request")
	}
	if err := manifest.Validate(); err != nil {
		return Content{}, err
	}
	for _, component := range manifest.Components {
		if component.Role != role || component.Name != name {
			continue
		}
		content, found, err := ReadContent(ctx, reader, component.Artifact)
		if err != nil {
			return Content{}, err
		}
		if !found {
			return Content{}, errors.New("artifact: manifest component content is absent")
		}
		return content, nil
	}
	return Content{}, errors.New("artifact: manifest component is absent")
}
