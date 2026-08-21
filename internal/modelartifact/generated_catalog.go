package modelartifact

import (
	"fmt"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tokenizer"
)

// ValidateGeneratedCatalog: exercise runtime admission against streamed output facts.
func ValidateGeneratedCatalog(metadata []gguf.Metadata, tensors []gguf.TensorData) error {
	file := &gguf.File{Metadata: metadata, Tensors: make([]gguf.TensorInfo, len(tensors))}
	for index, tensor := range tensors {
		info, err := gguf.NewTensorInfo(tensor.Name, tensor.Type, tensor.Shape)
		if err != nil {
			return err
		}
		file.Tensors[index] = info
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return fmt.Errorf("generated model metadata: %w", err)
	}
	if _, err := model.ReadWeights(file, spec); err != nil {
		return fmt.Errorf("generated tensor catalog: %w", err)
	}
	if _, err := tokenizer.Load(file); err != nil {
		return fmt.Errorf("generated tokenizer: %w", err)
	}
	return nil
}
