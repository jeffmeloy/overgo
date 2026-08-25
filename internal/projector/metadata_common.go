package projector

import (
	"errors"
	"fmt"
	"slices"
	"strconv"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensorcatalog"
)

func appendIndexedTensorNames(names []string, file *gguf.File, root string, count int, required, optional []string) []string {
	names = slices.Grow(names, count*(len(required)+len(optional)))
	for index := range count {
		prefix := root + "." + strconv.Itoa(index) + "."
		for _, suffix := range required {
			names = append(names, prefix+suffix)
		}
		for _, suffix := range optional {
			if name := prefix + suffix; hasTensor(file, name) {
				names = append(names, name)
			}
		}
	}
	return names
}

func appendPresentTensorNames(names []string, file *gguf.File, candidates ...string) []string {
	for _, name := range candidates {
		if hasTensor(file, name) {
			names = append(names, name)
		}
	}
	return names
}

func validateProjectorTensorNames(file *gguf.File, names []string) error {
	for _, name := range names {
		info, ok := file.Tensor(name)
		if !ok {
			return fmt.Errorf("projector: missing tensor %q", name)
		}
		if err := tensorcatalog.ValidateInfo(info, tensorcatalog.Requirement{
			Ranks: []uint32{tensor.SingletonExtent, tensor.PairedExtent, tensor.TripleExtent, tensor.MaxDimensions},
		}); err != nil {
			return fmt.Errorf("projector: tensor %q rank %d is invalid", name, info.Dimensions)
		}
	}
	return nil
}

func validateVisionProjector(file *gguf.File, typeKey, expectedType string) error {
	return validateProjector(file, typeKey, visionEncoderEnabledKey, expectedType, "vision")
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
