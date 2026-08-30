package modelartifact

import (
	"errors"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// FromSafetensorsPath inventories a weights-only Safetensors directory while
// preserving the source tensor names. It is the publication owner for model
// artifacts produced by offline composition, which intentionally have no
// copied Hugging Face configuration or companion files.
func FromSafetensorsPath(directory string) (Inventory, error) {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return Inventory{}, err
	}
	inventory, inventoryErr := fromSafetensorsSource(directory, source)
	return inventory, errors.Join(inventoryErr, source.Close())
}

func fromSafetensorsSource(directory string, source *safetensors.Source) (Inventory, error) {
	if source == nil || len(source.Tensors) == 0 {
		return Inventory{}, errors.New("model artifact: incomplete Safetensors source")
	}
	type componentSpec struct {
		path      string
		name      string
		role      artifact.ComponentRole
		kind      artifact.Kind
		mediaType string
	}
	shards := source.Shards()
	sharded := len(shards) > tensor.SingletonExtent
	weightRole := artifact.ComponentWeights
	if sharded {
		weightRole = artifact.ComponentWeightsShard
	}
	specs := make([]componentSpec, 0, len(shards)+tensor.SingletonExtent)
	if source.Indexed() {
		specs = append(specs, componentSpec{
			path: filepath.Join(directory, source.IndexName()), name: "shard-index",
			role: artifact.ComponentShardIndex, kind: artifact.KindFile, mediaType: jsonMediaType,
		})
	}
	for index, shard := range shards {
		name := "weights"
		if sharded {
			name = indexedName("weights", index)
		}
		specs = append(specs, componentSpec{
			path: filepath.Join(directory, shard), name: name, role: weightRole,
			kind: artifact.KindTensorSet, mediaType: safetensorsMediaType,
		})
	}
	roleOrdinals := map[artifact.ComponentRole]uint32{}
	components := make([]artifact.Component, 0, len(specs))
	descriptors := make([]artifact.Descriptor, 0, len(specs))
	locations := make([]artifact.Location, 0, len(specs)+tensor.SingletonExtent)
	for _, spec := range specs {
		descriptor, absolute, err := identifyFile(spec.path, spec.kind, spec.mediaType)
		if err != nil {
			return Inventory{}, fmt.Errorf("model artifact: Safetensors component %s: %w", spec.name, err)
		}
		ordinal := roleOrdinals[spec.role]
		roleOrdinals[spec.role]++
		components = append(components, artifact.Component{
			Role: spec.role, Ordinal: ordinal, Name: spec.name, Artifact: descriptor.ID,
		})
		descriptors = append(descriptors, descriptor)
		locations = append(locations, artifact.Location{
			Artifact: descriptor.ID, Kind: artifact.LocationFile, Value: absolute,
		})
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, components)
	if err != nil {
		return Inventory{}, err
	}
	manifestLocation, err := artifact.CanonicalLocalLocation(manifest.ID, artifact.LocationDirectory, directory)
	if err != nil {
		return Inventory{}, err
	}
	locations = append(locations, manifestLocation)
	tensors, err := NewSafetensorsTensorInventory(manifest.ID, source)
	if err != nil {
		return Inventory{}, err
	}
	return Inventory{
		Manifest: manifest, TensorInventory: tensors, Components: descriptors, Locations: locations,
	}, nil
}
