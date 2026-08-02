package projector

import (
	"fmt"
	"slices"

	"llamacpp2go/internal/gguf"
)

func validateProjectorTensorShapes(file *gguf.File, required map[string][]uint64) error {
	for name, shape := range required {
		info, ok := file.Tensor(name)
		if !ok {
			return fmt.Errorf("projector: missing tensor %q", name)
		}
		if int(info.Dimensions) != len(shape) {
			return fmt.Errorf("projector: tensor %q rank %d, want %d", name, info.Dimensions, len(shape))
		}
		if !slices.Equal(info.Shape[:info.Dimensions], shape) {
			return fmt.Errorf("projector: tensor %q shape %v, want %v", name, info.Shape[:info.Dimensions], shape)
		}
	}
	return nil
}
