package modelartifact

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/pathidentity"
	"overgo/internal/tensor"
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

// FromHFPath inventories one Hugging Face repository directory.
func FromHFPath(path string) (Inventory, error) {
	if _, err := os.Stat(filepath.Join(path, "model_index.json")); err == nil {
		return fromHFPipeline(path)
	} else if !errors.Is(err, os.ErrNotExist) {
		return Inventory{}, err
	}
	repository, err := hfrepo.Open(path)
	if err != nil {
		return Inventory{}, err
	}
	inventory, inventoryErr := FromHFRepository(repository)
	return inventory, errors.Join(inventoryErr, repository.Close())
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
		Lineage: i.TensorInventory.Lineage(),
	}, nil
}

// FromGGUFPath inventories all physical shards declared by a GGUF input.
func FromGGUFPath(path string) (Inventory, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return Inventory{}, err
	}
	inventory, inventoryErr := FromGGUF(file, artifact.KindModel)
	return inventory, errors.Join(inventoryErr, file.Close())
}

// FromGGUF: typed inventory from an already-validated logical GGUF.
func FromGGUF(file *gguf.File, kind artifact.Kind) (Inventory, error) {
	inventory, _, err := FromGGUFClaimed(file, kind, nil)
	return inventory, err
}

// Claimer reports the identity remembered for the file at an absolute path and
// the size it had, without reading the file.
type Claimer func(path string) (artifact.ID, uint64, bool)

// describeShard names a shard's identity from the claimer when it has one of
// the size the parsed file has, and by hashing the file otherwise.
func describeShard(path string, size uint64, claimer Claimer) (descriptor artifact.Descriptor, absolute string, claimed bool, err error) {
	if claimer != nil {
		if absolute, err = pathidentity.Canonical(path); err == nil {
			if id, remembered, ok := claimer(absolute); ok && remembered == size && id.Kind() == artifact.KindTensorSet {
				return artifact.Descriptor{ID: id, Size: size, MediaType: ggufMediaType}, absolute, true, nil
			}
		}
	}
	descriptor, absolute, err = identifyFile(path, artifact.KindTensorSet, ggufMediaType)
	return descriptor, absolute, false, err
}

// FromGGUFClaimed is FromGGUF where a shard the claimer knows is not read: its
// identity is a claim, and when any shard is claimed the shard identities are
// returned in shard order as claims the bytes have not yet proved. The caller
// must prove them before the model serves, by VerifyClaims or by a pass that
// hashes the bytes it uploads. The claims are nil when every shard was hashed.
func FromGGUFClaimed(file *gguf.File, kind artifact.Kind, claimer Claimer) (Inventory, []artifact.ID, error) {
	if file == nil {
		return Inventory{}, nil, errors.New("model artifact: nil GGUF")
	}
	if !tensorBearingKind(kind) {
		return Inventory{}, nil, errors.New("model artifact: GGUF logical kind must be model or projector")
	}
	paths := file.SourcePaths()
	if len(paths) == 0 {
		return Inventory{}, nil, errors.New("model artifact: GGUF has no physical source paths")
	}
	components := make([]artifact.Component, 0, len(paths))
	descriptors := make([]artifact.Descriptor, 0, len(paths))
	locations := make([]artifact.Location, 0, len(paths)+1)
	sharded := len(paths) > tensor.SingletonExtent
	role := artifact.ComponentWeights
	if sharded {
		role = artifact.ComponentWeightsShard
	}
	shards := file.Shards()
	var claims []artifact.ID
	anyClaimed := false
	for index, path := range paths {
		descriptor, absolute, claimed, err := describeShard(path, shards[index].Size, claimer)
		if err != nil {
			return Inventory{}, nil, fmt.Errorf("model artifact: GGUF component %d: %w", index, err)
		}
		anyClaimed = anyClaimed || claimed
		claims = append(claims, descriptor.ID)
		name := "weights"
		if sharded {
			name = indexedName("weights", index)
		}
		components = append(components, artifact.Component{
			Role: role, Ordinal: uint32(index), Name: name, Artifact: descriptor.ID,
		})
		descriptors = append(descriptors, descriptor)
		locations = append(locations, artifact.Location{Artifact: descriptor.ID, Kind: artifact.LocationFile, Value: absolute})
	}
	manifest, err := artifact.NewManifest(kind, components)
	if err != nil {
		return Inventory{}, nil, err
	}
	root := filepath.Dir(locations[0].Value)
	for _, location := range locations[1:] {
		if filepath.Dir(location.Value) != root {
			return Inventory{}, nil, errors.New("model artifact: GGUF shards span directories")
		}
	}
	manifestLocation, err := artifact.CanonicalLocalLocation(manifest.ID, artifact.LocationDirectory, root)
	if err != nil {
		return Inventory{}, nil, err
	}
	locations = append(locations, manifestLocation)
	tensors, err := NewGGUFTensorInventory(manifest.ID, file)
	if err != nil {
		return Inventory{}, nil, err
	}
	if !anyClaimed {
		claims = nil
	}
	return Inventory{Manifest: manifest, TensorInventory: tensors, Components: descriptors, Locations: locations}, claims, nil
}

// VerifyClaims proves claimed shard identities the way FromGGUF would have
// found them: by hashing each file. It is the whole-file pass a claim is spared
// when a load hashes the bytes it uploads.
func VerifyClaims(file *gguf.File, claims []artifact.ID) error {
	paths := file.SourcePaths()
	if len(paths) != len(claims) {
		return fmt.Errorf("model artifact: %d claimed identities for %d shards", len(claims), len(paths))
	}
	for index, path := range paths {
		descriptor, _, err := identifyFile(path, claims[index].Kind(), ggufMediaType)
		if err != nil {
			return fmt.Errorf("model artifact: GGUF component %d: %w", index, err)
		}
		if descriptor.ID != claims[index] {
			return fmt.Errorf("%w: %s is %s, claimed %s", artifact.ErrContentMismatch, path, descriptor.ID, claims[index])
		}
	}
	return nil
}

func tensorBearingKind(kind artifact.Kind) bool {
	return kind == artifact.KindModel || kind == artifact.KindProjector
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
	sharded := len(shards) > tensor.SingletonExtent
	weightRole := artifact.ComponentWeights
	if sharded {
		weightRole = artifact.ComponentWeightsShard
	}
	for index, name := range shards {
		logicalName := "weights"
		if sharded {
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
	manifestLocation, err := artifact.CanonicalLocalLocation(manifest.ID, artifact.LocationDirectory, repository.Directory)
	if err != nil {
		return Inventory{}, err
	}
	locations = append(locations, manifestLocation)
	tensors, err := NewSafetensorsTensorInventory(manifest.ID, repository.Tensors)
	if err != nil {
		return Inventory{}, err
	}
	return Inventory{Manifest: manifest, TensorInventory: tensors, Components: descriptors, Locations: locations}, nil
}

func identifyFile(path string, kind artifact.Kind, mediaType string) (artifact.Descriptor, string, error) {
	absolute, err := pathidentity.Canonical(path)
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
