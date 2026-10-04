package modelartifact

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/overgodb"
)

const (
	fixtureTensorName = "weight"
	fixtureConfig     = `{"model_type":"fixture"}`
	fixtureTokenizer  = `{"version":"1.0"}`
)

func TestGGUFInventoryUsesPhysicalContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	err = gguf.Write(file, nil, []gguf.TensorData{{
		Name: fixtureTensorName, Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, 4)),
	}}, gguf.WriteOptions{})
	closeErr := file.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	opened, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	inventory, err := FromGGUF(opened, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Components) != 1 || len(inventory.Manifest.Components) != 1 ||
		inventory.Manifest.Components[0].Role != artifact.ComponentWeights {
		t.Fatalf("inventory = %+v", inventory)
	}
	if inventory.Locations[0].Value != path {
		t.Fatalf("component location = %q, want %q", inventory.Locations[0].Value, path)
	}
	tensor, ok := inventory.TensorInventory.Tensor(fixtureTensorName)
	if !ok || tensor.Storage != "f32" || tensor.Bytes != 4 || !slices.Equal(tensor.Shape, []uint64{1}) {
		t.Fatalf("tensor fact = (%+v, %v)", tensor, ok)
	}
}

func TestHFInventoryIsRelocationStable(t *testing.T) {
	first := writeHFRepository(t)
	second := writeHFRepository(t)
	firstRepository, err := hfrepo.Open(first)
	if err != nil {
		t.Fatal(err)
	}
	defer firstRepository.Close()
	secondRepository, err := hfrepo.Open(second)
	if err != nil {
		t.Fatal(err)
	}
	defer secondRepository.Close()
	firstInventory, err := FromHFRepository(firstRepository)
	if err != nil {
		t.Fatal(err)
	}
	secondInventory, err := FromHFRepository(secondRepository)
	if err != nil {
		t.Fatal(err)
	}
	if firstInventory.Manifest.ID != secondInventory.Manifest.ID {
		t.Fatalf("relocated identities differ: %s != %s", firstInventory.Manifest.ID, secondInventory.Manifest.ID)
	}
	if firstInventory.TensorInventory.ID != secondInventory.TensorInventory.ID {
		t.Fatalf("relocated tensor inventories differ: %s != %s", firstInventory.TensorInventory.ID, secondInventory.TensorInventory.ID)
	}
	if len(firstInventory.Components) != 3 || len(firstInventory.Locations) != 4 {
		t.Fatalf("components/locations = %d/%d", len(firstInventory.Components), len(firstInventory.Locations))
	}
	batch, err := firstInventory.Batch("fixture/hf/import")
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Manifests) != 1 || len(batch.Contents) != 1 || len(batch.Lineage) != 1 ||
		len(batch.Locations) != len(firstInventory.Locations) {
		t.Fatalf("batch = %+v", batch)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	loaded, ok, err := LoadTensorInventory(t.Context(), store, firstInventory.Manifest.ID)
	if err != nil || !ok || loaded.ID != firstInventory.TensorInventory.ID {
		t.Fatalf("stored tensor inventory = (%s, %v, %v)", loaded.ID, ok, err)
	}
}

func TestHFShardedInventoryUsesOrderedLogicalSlots(t *testing.T) {
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(fixtureConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	shards := []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	writeSafetensor(t, filepath.Join(directory, shards[0]), "first")
	writeSafetensor(t, filepath.Join(directory, shards[1]), "second")
	index, err := json.Marshal(map[string]any{
		"weight_map": map[string]string{"first": shards[0], "second": shards[1]},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "model.safetensors.index.json"), index, 0o600); err != nil {
		t.Fatal(err)
	}
	repository, err := hfrepo.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	inventory, err := FromHFRepository(repository)
	if err != nil {
		t.Fatal(err)
	}
	var weights []artifact.Component
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentWeightsShard {
			weights = append(weights, component)
		}
	}
	if len(weights) != len(shards) || weights[0].Name != "weights/00001" || weights[1].Name != "weights/00002" {
		t.Fatalf("weight components = %+v", weights)
	}
}

func writeHFRepository(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(fixtureConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(fixtureTokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
	writeSafetensor(t, filepath.Join(directory, "model.safetensors"), fixtureTensorName)
	return directory
}

func writeSafetensor(t *testing.T, path, name string) {
	t.Helper()
	header := fmt.Sprintf(`{"%s":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, name)
	data := make([]byte, 8, 8+len(header)+1)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	data = append(data, 1)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// writeGGUFFixture writes a one-file model and returns its path.
func writeGGUFFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.gguf")
	err := gguf.WriteFileExclusive(path, nil, []gguf.TensorData{{
		Name: fixtureTensorName, Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, 4)),
	}}, gguf.WriteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	return path
}

// writeSplitGGUFFixture writes a model of four tensors as two split files and
// returns the path of the first.
func writeSplitGGUFFixture(t *testing.T) string {
	t.Helper()
	var whole bytes.Buffer
	var tensors []gguf.TensorData
	for index := range 4 {
		tensors = append(tensors, gguf.TensorData{
			Name: fmt.Sprintf("weight.%d", index), Shape: []uint64{4}, Type: gguf.DTypeF32,
			Data: bytes.NewReader(bytes.Repeat([]byte{byte(index + 1)}, 16)),
		})
	}
	if err := gguf.Write(&whole, nil, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	source, err := gguf.Parse(bytes.NewReader(whole.Bytes()), uint64(whole.Len()), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	prefix := filepath.Join(t.TempDir(), "split")
	name := func(index, count uint16) string { return fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, index+1, count) }
	err = source.WriteSplit(func(index, count uint16) (io.WriteCloser, error) {
		return os.Create(name(index, count))
	}, gguf.SplitOptions{MaxTensors: 2})
	if err != nil {
		t.Fatal(err)
	}
	return name(0, 2)
}

// TestFromGGUFClaimedTakesFreshClaimsWithoutReadingTheFile holds the claimed
// inventory to the hashed one when the claim is true, and shows the bytes were
// not read by giving it a claim they do not carry: the identity follows the
// claim. A claim of the wrong size or the wrong kind is refused, and its
// shard hashed.
func TestFromGGUFClaimedTakesFreshClaimsWithoutReadingTheFile(t *testing.T) {
	for name, path := range map[string]string{"single": writeGGUFFixture(t), "split": writeSplitGGUFFixture(t)} {
		t.Run(name, func(t *testing.T) {
			file, err := gguf.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			hashed, err := FromGGUF(file, artifact.KindModel)
			if err != nil {
				t.Fatal(err)
			}
			truth := map[string]artifact.Descriptor{}
			for index, location := range hashed.Locations[:len(hashed.Components)] {
				truth[location.Value] = hashed.Components[index]
			}
			claimerFor := func(change func(path string, descriptor artifact.Descriptor) (artifact.Descriptor, bool)) Claimer {
				return func(path string) (artifact.ID, uint64, bool) {
					descriptor, claimed := change(path, truth[path])
					return descriptor.ID, descriptor.Size, claimed
				}
			}
			same := func(_ string, descriptor artifact.Descriptor) (artifact.Descriptor, bool) { return descriptor, true }

			inventory, claims, err := FromGGUFClaimed(file, artifact.KindModel, claimerFor(same))
			if err != nil {
				t.Fatal(err)
			}
			if inventory.Manifest.ID != hashed.Manifest.ID || len(claims) != len(hashed.Components) ||
				!slices.Equal(inventory.Components, hashed.Components) || !slices.Equal(inventory.Locations, hashed.Locations) {
				t.Fatalf("claimed inventory differs from the hashed one: %+v vs %+v", inventory, hashed)
			}
			for index, component := range hashed.Components {
				if claims[index] != component.ID {
					t.Fatalf("claim %d = %s, want %s", index, claims[index], component.ID)
				}
			}

			invented, _ := artifact.IdentifyBytes(artifact.KindTensorSet, []byte("not these bytes"))
			followed, followedClaims, err := FromGGUFClaimed(file, artifact.KindModel, claimerFor(
				func(_ string, descriptor artifact.Descriptor) (artifact.Descriptor, bool) {
					descriptor.ID = invented
					return descriptor, true
				}))
			if err != nil || followed.Manifest.ID == hashed.Manifest.ID || followedClaims[0] != invented {
				t.Fatalf("an invented claim was not followed: manifest %s, claims %v, err %v", followed.Manifest.ID, followedClaims, err)
			}

			// A claim for the first shard only: the others are hashed, and every
			// shard's identity is returned for the load to prove.
			partial, partialClaims, err := FromGGUFClaimed(file, artifact.KindModel, claimerFor(
				func(candidate string, descriptor artifact.Descriptor) (artifact.Descriptor, bool) {
					return descriptor, candidate == hashed.Locations[0].Value
				}))
			if err != nil || partial.Manifest.ID != hashed.Manifest.ID || len(partialClaims) != len(hashed.Components) {
				t.Fatalf("partial claim: claims %v, manifest %s, err %v", partialClaims, partial.Manifest.ID, err)
			}

			model, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("kind"))
			refused := map[string]Claimer{
				"nil claimer": nil,
				"unclaimed":   claimerFor(func(string, artifact.Descriptor) (artifact.Descriptor, bool) { return artifact.Descriptor{}, false }),
				"wrong size": claimerFor(func(_ string, descriptor artifact.Descriptor) (artifact.Descriptor, bool) {
					descriptor.Size++
					return descriptor, true
				}),
				"wrong kind": claimerFor(func(_ string, descriptor artifact.Descriptor) (artifact.Descriptor, bool) {
					descriptor.ID = model
					return descriptor, true
				}),
			}
			for label, claimer := range refused {
				inventory, claims, err := FromGGUFClaimed(file, artifact.KindModel, claimer)
				if err != nil || claims != nil || inventory.Manifest.ID != hashed.Manifest.ID {
					t.Fatalf("%s: claims %v, manifest %s, err %v; want the hashed inventory", label, claims, inventory.Manifest.ID, err)
				}
			}
		})
	}
}

// TestVerifyClaimsHashesEveryShard proves true claims and refuses a false one,
// a claim for the wrong number of shards, and a changed file.
func TestVerifyClaimsHashesEveryShard(t *testing.T) {
	path := writeSplitGGUFFixture(t)
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	var claims []artifact.ID
	for _, component := range inventory.Components {
		claims = append(claims, component.ID)
	}
	if err := VerifyClaims(file, claims); err != nil {
		t.Fatal(err)
	}
	if err := VerifyClaims(file, claims[:1]); err == nil {
		t.Fatal("verified one claim for two shards")
	}
	swapped := []artifact.ID{claims[1], claims[0]}
	if err := VerifyClaims(file, swapped); !errors.Is(err, artifact.ErrContentMismatch) {
		t.Fatalf("verified swapped claims: %v", err)
	}
	second := file.SourcePaths()[1]
	data, err := os.ReadFile(second)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-1] ^= 1
	if err := os.WriteFile(second, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := VerifyClaims(file, claims); !errors.Is(err, artifact.ErrContentMismatch) {
		t.Fatalf("verified claims for a changed file: %v", err)
	}
}
