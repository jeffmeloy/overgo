package modelartifact

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/safetensors"
)

// FileSpec: one explicit inventory component.
type FileSpec struct {
	Path string // component file path
	Name string // logical component name; also the tensor-fact namespace for weights
	Role artifact.ComponentRole
}

// FromFiles builds an Inventory from an explicit component list, for
// artifacts whose directory layout no repository walker owns (e.g. a
// multi-sub-model directory holding one safetensors file per head).
// Safetensors weights get tensor facts namespaced "<name>/<tensor>" so
// heads with identical tensor names stay distinct; pytorch-zip weights
// (.pth/.pt) are content-hashed components without fact extraction.
func FromFiles(directory string, specs []FileSpec) (Inventory, error) {
	if directory == "" || len(specs) == 0 {
		return Inventory{}, errors.New("model artifact: file inventory requires a directory and components")
	}
	roleOrdinals := map[artifact.ComponentRole]uint32{}
	components := make([]artifact.Component, 0, len(specs))
	descriptors := make([]artifact.Descriptor, 0, len(specs))
	locations := make([]artifact.Location, 0, len(specs)+1)
	var facts []TensorFact
	for _, spec := range specs {
		kind, mediaType := fileContract(spec.Role, spec.Path)
		descriptor, absolute, err := identifyFile(spec.Path, kind, mediaType)
		if err != nil {
			return Inventory{}, fmt.Errorf("model artifact: component %s: %w", spec.Name, err)
		}
		ordinal := roleOrdinals[spec.Role]
		roleOrdinals[spec.Role]++
		components = append(components, artifact.Component{
			Role: spec.Role, Ordinal: ordinal, Name: spec.Name, Artifact: descriptor.ID,
		})
		descriptors = append(descriptors, descriptor)
		locations = append(locations, artifact.Location{Artifact: descriptor.ID, Kind: artifact.LocationFile, Value: absolute})
		if mediaType == safetensorsMediaType {
			weightFacts, err := namespacedTensorFacts(absolute, spec.Name)
			if err != nil {
				return Inventory{}, fmt.Errorf("model artifact: component %s: %w", spec.Name, err)
			}
			facts = append(facts, weightFacts...)
		}
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, components)
	if err != nil {
		return Inventory{}, err
	}
	absoluteRoot, err := filepath.Abs(directory)
	if err != nil {
		return Inventory{}, err
	}
	locations = append(locations, artifact.Location{
		Artifact: manifest.ID, Kind: artifact.LocationDirectory, Value: filepath.Clean(absoluteRoot),
	})
	sort.Slice(facts, func(left, right int) bool { return facts[left].Name < facts[right].Name })
	tensors, err := NewTensorInventoryDocument(manifest.ID, TensorFormatSafetensors, facts)
	if err != nil {
		return Inventory{}, err
	}
	return Inventory{Manifest: manifest, TensorInventory: tensors, Components: descriptors, Locations: locations}, nil
}

func fileContract(role artifact.ComponentRole, path string) (artifact.Kind, string) {
	switch role {
	case artifact.ComponentConfig:
		return artifact.KindFile, jsonMediaType
	case artifact.ComponentWeights, artifact.ComponentWeightsShard:
		// Media type follows the container: safetensors gets tensor-fact
		// extraction; pytorch-zip (.pth/.pt) is a tensor set whose facts
		// are not parsed here (internal/pytorchzip owns that reader).
		switch strings.ToLower(filepath.Ext(path)) {
		case ".pth", ".pt":
			return artifact.KindTensorSet, pytorchZipMediaType
		}
		return artifact.KindTensorSet, safetensorsMediaType
	default:
		return artifact.KindFile, binaryMediaType
	}
}

// namespacedTensorFacts reads one safetensors file's logical facts. The
// source contract is directory-scoped, so the parent open must resolve to
// exactly this file.
func namespacedTensorFacts(path, namespace string) ([]TensorFact, error) {
	source, err := safetensors.OpenSource(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	defer source.Close()
	if shards := source.Shards(); len(shards) != 1 || shards[0] != filepath.Base(path) {
		return nil, fmt.Errorf("weights file %s does not stand alone in its directory (shards %v)", filepath.Base(path), source.Shards())
	}
	names := source.Names()
	facts := make([]TensorFact, len(names))
	for index, name := range names {
		tensor := source.Tensors[name]
		facts[index] = TensorFact{
			Name: namespace + "/" + name, Shape: slices.Clone(tensor.Shape),
			Storage: strings.ToLower(tensor.DType), Bytes: uint64(tensor.Size()),
		}
	}
	return facts, nil
}
