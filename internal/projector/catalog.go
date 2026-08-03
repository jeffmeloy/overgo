package projector

import (
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensorcatalog"
)

func validateProjectorTensorShapes(file *gguf.File, required map[string][]uint64) error {
	tensors := make(map[string]gguf.TensorInfo, len(required))
	requirements := make([]tensorcatalog.Requirement, 0, len(required))
	for name, shape := range required {
		if info, ok := file.Tensor(name); ok {
			tensors[name] = info
		}
		requirements = append(requirements, tensorcatalog.Requirement{Name: name, Shape: shape})
	}
	if err := tensorcatalog.Validate(tensors, "", requirements); err != nil {
		return fmt.Errorf("projector: %w", err)
	}
	return nil
}
