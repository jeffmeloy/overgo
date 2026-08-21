package modelartifact

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/pytorchzip"
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
func FromFiles(directory string, specs []FileSpec) (inventory Inventory, err error) {
	if directory == "" || len(specs) == 0 {
		return Inventory{}, errors.New("model artifact: file inventory requires a directory and components")
	}
	roleOrdinals := map[artifact.ComponentRole]uint32{}
	components := make([]artifact.Component, 0, len(specs))
	descriptors := make([]artifact.Descriptor, 0, len(specs))
	locations := make([]artifact.Location, 0, len(specs)+1)
	sources := make(map[string]*safetensors.Source)
	defer func() {
		for _, source := range sources {
			err = errors.Join(err, source.Close())
		}
	}()
	var facts []TensorFact
	hasSafetensors, hasPyTorch := false, false
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
			hasSafetensors = true
			sourceDir := filepath.Dir(absolute)
			source := sources[sourceDir]
			if source == nil {
				source, err = safetensors.OpenSource(sourceDir)
				if err != nil {
					return Inventory{}, err
				}
				sources[sourceDir] = source
			}
			weightFacts, err := namespacedTensorFacts(source, absolute, spec.Name)
			if err != nil {
				return Inventory{}, fmt.Errorf("model artifact: component %s: %w", spec.Name, err)
			}
			facts = append(facts, weightFacts...)
		} else if mediaType == pytorchZipMediaType {
			hasPyTorch = true
			weightFacts, err := pytorchTensorFacts(absolute, spec.Name)
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
	format := TensorFormatSafetensors
	if hasPyTorch {
		format = TensorFormatPyTorch
		if hasSafetensors {
			format = TensorFormatMixed
		}
	}
	tensors, err := NewTensorInventoryDocument(manifest.ID, format, facts)
	if err != nil {
		return Inventory{}, err
	}
	return Inventory{Manifest: manifest, TensorInventory: tensors, Components: descriptors, Locations: locations}, nil
}

func pytorchTensorFacts(filename, namespace string) ([]TensorFact, error) {
	catalog, err := pytorchzip.ReadCatalog(filename)
	if err != nil {
		return nil, err
	}
	facts := make([]TensorFact, len(catalog.Tensors))
	for index, tensor := range catalog.Tensors {
		storage, err := pytorchzip.StorageType(tensor.DType)
		if err != nil {
			return nil, fmt.Errorf("tensor %s: %w", tensor.Name, err)
		}
		elements, ok := checked.Uint64(tensor.Numel)
		if !ok {
			return nil, fmt.Errorf("tensor %s: negative element count", tensor.Name)
		}
		traits, _ := storage.Traits()
		bytes, ok := checked.Bytes(elements, traits.TypeSize)
		if !ok {
			return nil, fmt.Errorf("tensor %s: byte size overflows", tensor.Name)
		}
		shape := make([]uint64, len(tensor.Shape))
		for dimension, size := range tensor.Shape {
			extent, ok := checked.Uint64(size)
			if !ok {
				return nil, fmt.Errorf("tensor %s: negative dimension", tensor.Name)
			}
			shape[dimension] = extent
		}
		facts[index] = TensorFact{
			Name: namespace + "/" + tensor.Name, Shape: shape,
			Storage: storage.String(), Bytes: bytes,
		}
	}
	return facts, nil
}

func fileContract(role artifact.ComponentRole, path string) (artifact.Kind, string) {
	switch role {
	case artifact.ComponentConfig, artifact.ComponentShardIndex:
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
func namespacedTensorFacts(source *safetensors.Source, path, namespace string) ([]TensorFact, error) {
	base := filepath.Base(path)
	if !source.ContainsOnlyShard(base) && (!source.Indexed() || !source.ContainsShard(base)) {
		return nil, fmt.Errorf("weights file %s does not stand alone in its directory (shards %v)", base, source.Shards())
	}
	var names []string
	for _, name := range source.Names() {
		if source.Tensors[name].Shard == base {
			names = append(names, name)
		}
	}
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
