package projector

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
)

func validateVisionProjector(file *gguf.File, typeKey, expectedType string) error {
	return validateProjector(file, typeKey, "clip.has_vision_encoder", expectedType, "vision")
}

func validateProjector(file *gguf.File, typeKey, enabledKey, expectedType, modality string) error {
	if file == nil {
		return errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return err
	}
	projectorType, err := metadataString(file, typeKey)
	if err != nil {
		return err
	}
	if architecture != "clip" || projectorType != expectedType {
		return fmt.Errorf(
			"projector: architecture/type %q/%q is not clip/%s",
			architecture, projectorType, expectedType,
		)
	}
	enabled, err := metadataBool(file, enabledKey)
	if err != nil {
		return err
	}
	if !enabled {
		return fmt.Errorf("projector: %s encoder is disabled", modality)
	}
	return nil
}
