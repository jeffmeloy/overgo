// Package gguf reads and writes GGUF, the file a model artifact is.
package gguf

import "fmt"

// Catalog builds an in-memory file over a metadata and tensor catalog with
// the lookups a read file has, so a converter can hand what it is about to
// write to the readers that will consume it before any byte lands. Duplicate
// keys and names are refused as the reader refuses them.
func Catalog(metadata []Metadata, tensors []TensorInfo) (*File, error) {
	file := &File{
		Version: CurrentVersion, Alignment: DefaultAlignment, Metadata: metadata, Tensors: tensors,
		metadataByKey: make(map[string]int, len(metadata)), tensorByName: make(map[string]int, len(tensors)),
	}
	for index, item := range metadata {
		if item.Key == "" {
			return nil, fmt.Errorf("gguf catalog: metadata key %d is empty", index)
		}
		if previous, exists := file.metadataByKey[item.Key]; exists {
			return nil, fmt.Errorf("gguf catalog: duplicate metadata key %q at indexes %d and %d", item.Key, previous, index)
		}
		file.metadataByKey[item.Key] = index
	}
	for index, tensor := range tensors {
		if tensor.Name == "" {
			return nil, fmt.Errorf("gguf catalog: tensor %d has no name", index)
		}
		if previous, exists := file.tensorByName[tensor.Name]; exists {
			return nil, fmt.Errorf("gguf catalog: duplicate tensor %q at indexes %d and %d", tensor.Name, previous, index)
		}
		file.tensorByName[tensor.Name] = index
	}
	return file, nil
}
