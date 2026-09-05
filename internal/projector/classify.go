package projector

import (
	"overgo/internal/gguf"
)

// DescribesProjection reports whether a GGUF file declares itself a
// projector: its own projector-type metadata decides, never its name, so a
// download directory holding a model beside its projector is sorted by
// what each file says it is. Whether the declared type is one the catalog
// supports is the registration's question, not this one.
func DescribesProjection(path string) (bool, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	for _, key := range []string{visionProjectorTypeKey, visionTowerTypeKey, audioProjectorTypeKey} {
		if value, ok := file.MetadataValue(key); ok && value.Type == gguf.ValueTypeString {
			if projectorType, _ := value.Data.(string); projectorType != "" {
				return true, nil
			}
		}
	}
	return false, nil
}
