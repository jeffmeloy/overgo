package modelartifact

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
)

const (
	ggufMediaType        = "application/vnd.gguf"
	safetensorsMediaType = "application/vnd.safetensors"
	pytorchZipMediaType  = "application/vnd.pytorch.zip"
	jsonMediaType        = artifact.JSONMediaType
	textMediaType        = "text/plain"
	binaryMediaType      = "application/octet-stream"
	componentDigits      = 5
)

// Inventory: logical model plus physical component facts
type Inventory struct {
	Manifest        artifact.Manifest
	TensorInventory TensorInventoryDocument
	Components      []artifact.Descriptor
	Locations       []artifact.Location
}

func (i Inventory) Batch(key string) (artifact.Batch, error) {
	tensors, err := i.TensorInventory.Content()
	if err != nil {
		return artifact.Batch{}, err
	}
	locations := make([]artifact.LocationEvent, len(i.Locations))
	for index, location := range i.Locations {
		locations[index] = artifact.LocationEvent{Location: location, Action: artifact.LocationAdd}
	}
	return artifact.Batch{
		Key: key, Artifacts: i.Components, Contents: []artifact.Content{tensors},
		Manifests: []artifact.Manifest{i.Manifest}, Locations: locations,
		Lineage: []artifact.Lineage{{
			Child: i.TensorInventory.ID, Parent: i.Manifest.ID, Relation: artifact.RelationDerivedFrom,
		}},
	}, nil
}

// FromGGUF: inventory from an already-validated logical GGUF
func FromGGUF(file *gguf.File) (Inventory, error) {
	if file == nil {
		return Inventory{}, errors.New("model artifact: nil GGUF")
	}
	paths := file.SourcePaths()
	if len(paths) == 0 {
		return Inventory{}, errors.New("model artifact: GGUF has no physical source paths")
	}
	components := make([]artifact.Component, 0, len(paths))
	descriptors := make([]artifact.Descriptor, 0, len(paths))
	locations := make([]artifact.Location, 0, len(paths)+1)
	role := artifact.ComponentWeights
	if len(paths) > 1 {
		role = artifact.ComponentWeightsShard
	}
	for index, path := range paths {
		descriptor, absolute, err := identifyFile(path, artifact.KindTensorSet, ggufMediaType)
		if err != nil {
			return Inventory{}, fmt.Errorf("model artifact: GGUF component %d: %w", index, err)
		}
		name := "weights"
		if len(paths) > 1 {
			name = indexedName("weights", index)
		}
		components = append(components, artifact.Component{
			Role: role, Ordinal: uint32(index), Name: name, Artifact: descriptor.ID,
		})
		descriptors = append(descriptors, descriptor)
		locations = append(locations, artifact.Location{Artifact: descriptor.ID, Kind: artifact.LocationFile, Value: absolute})
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, components)
	if err != nil {
		return Inventory{}, err
	}
	root := filepath.Dir(locations[0].Value)
	for _, location := range locations[1:] {
		if filepath.Dir(location.Value) != root {
			return Inventory{}, errors.New("model artifact: GGUF shards span directories")
		}
	}
	locations = append(locations, artifact.Location{Artifact: manifest.ID, Kind: artifact.LocationDirectory, Value: root})
	tensors, err := NewGGUFTensorInventory(manifest.ID, file)
	if err != nil {
		return Inventory{}, err
	}
	return Inventory{Manifest: manifest, TensorInventory: tensors, Components: descriptors, Locations: locations}, nil
}

// FromHFRepository: inventory from an already-validated Safetensors repository
func FromHFRepository(repository *hfrepo.Repository) (Inventory, error) {
	if repository == nil || repository.Tensors == nil || repository.Directory == "" {
		return Inventory{}, errors.New("model artifact: incomplete Hugging Face repository")
	}
	type fileSpec struct {
		path      string
		name      string
		role      artifact.ComponentRole
		kind      artifact.Kind
		mediaType string
	}
	specs := []fileSpec{{
		path: filepath.Join(repository.Directory, "config.json"), name: "config",
		role: artifact.ComponentConfig, kind: artifact.KindFile, mediaType: jsonMediaType,
	}}
	indexPath := filepath.Join(repository.Directory, "model.safetensors.index.json")
	if info, err := os.Stat(indexPath); err == nil && !info.IsDir() {
		specs = append(specs, fileSpec{
			path: indexPath, name: "shard-index", role: artifact.ComponentShardIndex,
			kind: artifact.KindFile, mediaType: jsonMediaType,
		})
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return Inventory{}, fmt.Errorf("model artifact: inspect shard index: %w", err)
	}
	shards := repository.Tensors.Shards()
	weightRole := artifact.ComponentWeights
	if len(shards) > 1 {
		weightRole = artifact.ComponentWeightsShard
	}
	for index, name := range shards {
		logicalName := "weights"
		if len(shards) > 1 {
			logicalName = indexedName("weights", index)
		}
		specs = append(specs, fileSpec{
			path: filepath.Join(repository.Directory, name), name: logicalName,
			role: weightRole, kind: artifact.KindTensorSet, mediaType: safetensorsMediaType,
		})
	}
	for _, name := range repository.Companions {
		role, kind, mediaType := companionContract(name)
		specs = append(specs, fileSpec{
			path: filepath.Join(repository.Directory, name), name: name,
			role: role, kind: kind, mediaType: mediaType,
		})
	}
	roleOrdinals := map[artifact.ComponentRole]uint32{}
	components := make([]artifact.Component, 0, len(specs))
	descriptors := make([]artifact.Descriptor, 0, len(specs))
	locations := make([]artifact.Location, 0, len(specs)+1)
	for _, spec := range specs {
		descriptor, absolute, err := identifyFile(spec.path, spec.kind, spec.mediaType)
		if err != nil {
			return Inventory{}, fmt.Errorf("model artifact: component %s: %w", spec.name, err)
		}
		ordinal := roleOrdinals[spec.role]
		roleOrdinals[spec.role]++
		components = append(components, artifact.Component{
			Role: spec.role, Ordinal: ordinal, Name: spec.name, Artifact: descriptor.ID,
		})
		descriptors = append(descriptors, descriptor)
		locations = append(locations, artifact.Location{Artifact: descriptor.ID, Kind: artifact.LocationFile, Value: absolute})
	}
	manifest, err := artifact.NewManifest(artifact.KindModel, components)
	if err != nil {
		return Inventory{}, err
	}
	locations = append(locations, artifact.Location{
		Artifact: manifest.ID, Kind: artifact.LocationDirectory, Value: filepath.Clean(repository.Directory),
	})
	tensors, err := NewSafetensorsTensorInventory(manifest.ID, repository.Tensors)
	if err != nil {
		return Inventory{}, err
	}
	return Inventory{Manifest: manifest, TensorInventory: tensors, Components: descriptors, Locations: locations}, nil
}

func identifyFile(path string, kind artifact.Kind, mediaType string) (artifact.Descriptor, string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return artifact.Descriptor{}, "", err
	}
	before, err := os.Stat(absolute)
	if err != nil {
		return artifact.Descriptor{}, "", err
	}
	if before.IsDir() {
		return artifact.Descriptor{}, "", errors.New("component is a directory")
	}
	file, err := os.Open(absolute)
	if err != nil {
		return artifact.Descriptor{}, "", err
	}
	id, size, hashErr := artifact.Identify(kind, file)
	closeErr := file.Close()
	if hashErr != nil {
		return artifact.Descriptor{}, "", hashErr
	}
	if closeErr != nil {
		return artifact.Descriptor{}, "", closeErr
	}
	after, err := os.Stat(absolute)
	if err != nil {
		return artifact.Descriptor{}, "", err
	}
	if before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || size != uint64(after.Size()) {
		return artifact.Descriptor{}, "", errors.New("component changed while hashing")
	}
	return artifact.Descriptor{ID: id, Size: size, MediaType: mediaType}, absolute, nil
}

func companionContract(name string) (artifact.ComponentRole, artifact.Kind, string) {
	switch name {
	case "chat_template.jinja":
		return artifact.ComponentChatTemplate, artifact.KindFile, textMediaType
	case "generation_config.json":
		return artifact.ComponentGenerationConfig, artifact.KindFile, jsonMediaType
	case "merges.txt":
		return artifact.ComponentMerges, artifact.KindTokenizer, textMediaType
	case "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json":
		return artifact.ComponentPreprocessor, artifact.KindFile, jsonMediaType
	case "vocab.json":
		return artifact.ComponentVocabulary, artifact.KindTokenizer, jsonMediaType
	case "added_tokens.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json":
		return artifact.ComponentTokenizer, artifact.KindTokenizer, jsonMediaType
	case "spiece.model", "tokenizer.model":
		return artifact.ComponentTokenizer, artifact.KindTokenizer, binaryMediaType
	default:
		return artifact.ComponentCompanion, artifact.KindFile, binaryMediaType
	}
}

func indexedName(prefix string, index int) string {
	return fmt.Sprintf("%s/%0*d", prefix, componentDigits, index+1)
}
