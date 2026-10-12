package mediacapability

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentvideo"
	"overgo/internal/overgodb"
	"overgo/internal/pathidentity"
	"overgo/internal/rae"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/trainingworkflow"
)

// torchStateDict is a protocol-2 torch state_dict holding one float32 weight
// of shape [1], whose four little-endian bytes are value: the smallest file
// the inventory reads as PyTorch weights.
func torchStateDict(t *testing.T, value string) []byte {
	t.Helper()
	metadata := "\x80\x02}X\x06\x00\x00\x00weightctorch._utils\n_rebuild_tensor_v2\n((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x01tQK\x00K\x01\x85K\x01\x85\x89)tRs."
	var encoded bytes.Buffer
	archive := zip.NewWriter(&encoded)
	for _, entry := range [][2]string{{"archive/data.pkl", metadata}, {"archive/data/0", value}} {
		writer, err := archive.CreateHeader(&zip.FileHeader{Name: entry[0], Method: zip.Store})
		if err == nil {
			_, err = writer.Write([]byte(entry[1]))
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	return encoded.Bytes()
}

// oneWeight is a safetensors file holding one float32 weight of shape [1].
func oneWeight(t *testing.T, value float32) []byte {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := safetensors.Save(path, map[string][]float32{"weight": {value}}, map[string][]int{"weight": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	encoded, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// The layout a collection is served from and the name of the encoder beside
// it, as internal/rae/recipes/sampler.json declares them; the package keeps
// them to itself.
const (
	pairCheckpoint = "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-S_ep20/stage2_model.pt"
	pairDecoder    = "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
	pairStatistics = "stats/dinov2/wReg_base/imagenet1k/stat.pt"
	pairOwnEncoder = "dinov2-with-registers-base"
)

// TestABridgedPairTrainsUnderTheModelItIsServedAs holds the model a training
// run records for a bridged pair to the model the pair is served as: every
// file serving names, and no other. A run is given the pair's bridge file, and
// trains and judges through the decoder, the latent statistics and both
// encoders' files beside it, so a same-shape replacement of any of them must
// change the model its recipe, checkpoint and verdict name, a missing one must
// refuse, and the same bytes under another root must be the same model.
func TestABridgedPairTrainsUnderTheModelItIsServedAs(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	models, err := pathidentity.Canonical(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	write := func(data []byte, parts ...string) string {
		t.Helper()
		path := filepath.Join(append([]string{models}, parts...)...)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	write([]byte("a checkpoint only looked for"), "collection", filepath.FromSlash(pairCheckpoint))
	decoder := write(torchStateDict(t, "\x00\x00\x80\x3f"), "collection", filepath.FromSlash(pairDecoder))
	statistics := write(torchStateDict(t, "\x00\x00\x00\x40"), "collection", filepath.FromSlash(pairStatistics))
	ownProcessor := write([]byte(`{"own":"processor"}`), pairOwnEncoder, "preprocessor_config.json")
	sourceWeights := write(oneWeight(t, 1), "other-encoder", "model.safetensors")
	sourceConfig := write([]byte(`{"source":"config"}`), "other-encoder", "config.json")
	sourceProcessor := write([]byte(`{"source":"processor"}`), "other-encoder", "preprocessor_config.json")
	pairDirectory := filepath.Join(models, "pair")
	if err := rae.SaveEncoderBridge(pairDirectory, rae.EncoderBridge{Weight: []float32{1, 0, 2, 0, -1, 0.5}, Bias: []float32{10, 20}}, "other-encoder", "collection"); err != nil {
		t.Fatal(err)
	}
	resolved, held, err := rae.ResolveBridgedPair(pairDirectory)
	if err != nil || !held {
		t.Fatalf("the saved pair did not resolve (%t, %v)", held, err)
	}
	bridge := resolved.Bridge

	component := func(role artifact.ComponentRole, ordinal uint32, name, path string) artifact.Component {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		kind := artifact.KindFile
		if role == artifact.ComponentWeights {
			kind = artifact.KindTensorSet
		}
		id, err := artifact.IdentifyBytes(kind, data)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Component{Role: role, Ordinal: ordinal, Name: name, Artifact: id}
	}
	served := func(directory string) artifact.ID {
		t.Helper()
		source, err := Catalog[recipe.TaskReconstruction].Resolve(directory)
		if err != nil {
			t.Fatalf("serving did not resolve %s: %v", directory, err)
		}
		return source.Inventory.Manifest.ID
	}
	trained := func(path string) artifact.ID {
		t.Helper()
		id, err := trainingworkflow.WeightsIdentity(path)
		if err != nil {
			t.Fatalf("training did not identify %s: %v", path, err)
		}
		return id
	}

	// Serving names the pair by these seven files, each by its bytes, and
	// records the pair's own directory beside the parent the files share. A
	// move of the layout that renamed or reordered one would shift the served
	// and the trained model together, so this holds the served one in place.
	source, err := Catalog[recipe.TaskReconstruction].Resolve(pairDirectory)
	if err != nil {
		t.Fatal(err)
	}
	want := []artifact.Component{
		component(artifact.ComponentWeights, 0, "bridge", bridge),
		component(artifact.ComponentWeights, 1, "decoder", decoder),
		component(artifact.ComponentWeights, 2, "statistics", statistics),
		component(artifact.ComponentWeights, 3, "encoder/weights", sourceWeights),
		component(artifact.ComponentConfig, 0, "encoder/config", sourceConfig),
		component(artifact.ComponentConfig, 1, "encoder/processor", sourceProcessor),
		component(artifact.ComponentConfig, 2, "decoder/processor", ownProcessor),
	}
	if !slices.Equal(source.Inventory.Manifest.Components, want) {
		t.Fatalf("the pair is served as\n%+v\nwant\n%+v", source.Inventory.Manifest.Components, want)
	}
	first := source.Inventory.Manifest.ID
	if source.Define == nil {
		t.Fatal("the served pair defines no recipe")
	}
	if definition, _, err := source.Define(first); err != nil || definition.Model != first || definition.Task != recipe.TaskReconstruction {
		t.Errorf("the served pair defines a %q recipe for model %s (%v), want reconstruction by %s", definition.Task, definition.Model, err, first)
	}
	for _, directory := range []string{models, pairDirectory} {
		if location := (artifact.Location{Artifact: first, Kind: artifact.LocationDirectory, Value: directory}); !slices.Contains(source.Inventory.Locations, location) {
			t.Errorf("the served pair records no directory %s: %+v", directory, source.Inventory.Locations)
		}
	}

	if got := trained(bridge); got != first {
		t.Errorf("a run over the pair's bridge file trains model %s, and the pair is served as %s", got, first)
	}
	if got := trained(pairDirectory); got != first {
		t.Errorf("a run over the pair's directory trains model %s, and the pair is served as %s", got, first)
	}

	// Each file the pair runs through, replaced by one of the same shape, is
	// another model to serving and to training alike, and none seen before.
	seen := map[artifact.ID]bool{first: true}
	for _, part := range []struct {
		name, path string
		changed    []byte
	}{
		{"decoder", decoder, torchStateDict(t, "\x00\x00\x40\x40")},
		{"latent statistics", statistics, torchStateDict(t, "\x00\x00\x80\x40")},
		{"source encoder's weights", sourceWeights, oneWeight(t, 5)},
		{"source encoder's configuration", sourceConfig, []byte(`{"source":"other config"}`)},
		{"source encoder's image processor", sourceProcessor, []byte(`{"source":"other processor"}`)},
		{"own encoder's image processor", ownProcessor, []byte(`{"own":"other processor"}`)},
	} {
		original, err := os.ReadFile(part.path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(part.path, part.changed, 0o600); err != nil {
			t.Fatal(err)
		}
		servedAs, trainedAs := served(pairDirectory), trained(bridge)
		if servedAs != trainedAs || seen[trainedAs] {
			t.Errorf("with another %s of the same shape the pair is served as %s and trains as %s, want one model not seen before", part.name, servedAs, trainedAs)
		}
		seen[servedAs], seen[trainedAs] = true, true
		if err := os.WriteFile(part.path, original, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	// What neither reads leaves the model as it was: the collection's DiT
	// checkpoint, and the own encoder's weights and configuration, of which
	// a bridged pair reads only the image processor. With every byte back the
	// model is the first one again.
	write([]byte("another checkpoint"), "collection", filepath.FromSlash(pairCheckpoint))
	write(oneWeight(t, 7), pairOwnEncoder, "model.safetensors")
	write([]byte(`{"own":"config"}`), pairOwnEncoder, "config.json")
	if servedAs, trainedAs := served(pairDirectory), trained(bridge); servedAs != first || trainedAs != first {
		t.Errorf("after files the pair does not read changed it is served as %s and trains as %s, want %s still", servedAs, trainedAs, first)
	}

	// The same bytes under another root are the same model.
	copied, err := pathidentity.Canonical(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.CopyFS(copied, os.DirFS(models)); err != nil {
		t.Fatal(err)
	}
	if servedAs, trainedAs := served(filepath.Join(copied, "pair")), trained(filepath.Join(copied, "pair", filepath.Base(bridge))); servedAs != first || trainedAs != first {
		t.Errorf("copied under another root the pair is served as %s and trains as %s, want %s", servedAs, trainedAs, first)
	}

	// A run's recipe names the served model, a second run finds the same
	// recipe, and the store can name the pair's own directory, the one the
	// model runs from. That holds whether or not serving recorded the pair
	// first under a key of its own.
	for _, registered := range []bool{false, true} {
		store, err := overgodb.Open(filepath.Join(t.TempDir(), "store"))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if registered {
			registration, err := source.Inventory.Batch("registered/" + first.String())
			if err != nil {
				t.Fatal(err)
			}
			if _, err := artifact.CommitBatch(ctx, store, registration); err != nil {
				t.Fatal(err)
			}
		}
		var recipes [2]artifact.ID
		for run := range recipes {
			if recipes[run], err = trainingworkflow.BootstrapRecipe(ctx, store, bridge, []byte("the pictures a run reads"), trainingworkflow.TokenContract); err != nil {
				t.Fatalf("bootstrap %d over the pair (served first: %t): %v", run, registered, err)
			}
		}
		definition, err := recipe.RequireDefinition(ctx, store, recipes[0])
		if err != nil {
			t.Fatal(err)
		}
		if definition.Model != first || recipes[1] != recipes[0] {
			t.Errorf("served first: %t: the run's recipe %s names model %s and a second run got recipe %s, want one recipe for model %s", registered, recipes[0], definition.Model, recipes[1], first)
		}
		locations, err := store.Locations(ctx, first)
		if err != nil {
			t.Fatal(err)
		}
		if own := (artifact.Location{Artifact: first, Kind: artifact.LocationDirectory, Value: pairDirectory}); !slices.Contains(locations, own) {
			t.Errorf("served first: %t: the store records no directory %s for the model the run trained: %+v", registered, pairDirectory, locations)
		}
	}

	// The collection's own pair is still served by its five files, the own
	// encoder's three among them.
	collection, err := Catalog[recipe.TaskReconstruction].Resolve(filepath.Join(models, "collection"))
	if err != nil {
		t.Fatal(err)
	}
	own := filepath.Join(models, pairOwnEncoder)
	if want := []artifact.Component{
		component(artifact.ComponentWeights, 0, "decoder", decoder),
		component(artifact.ComponentWeights, 1, "statistics", statistics),
		component(artifact.ComponentWeights, 2, "encoder/weights", filepath.Join(own, "model.safetensors")),
		component(artifact.ComponentConfig, 0, "encoder/config", filepath.Join(own, "config.json")),
		component(artifact.ComponentConfig, 1, "encoder/processor", ownProcessor),
	}; !slices.Equal(collection.Inventory.Manifest.Components, want) {
		t.Errorf("the collection is served as\n%+v\nwant\n%+v", collection.Inventory.Manifest.Components, want)
	}
	// A directory that is neither is still read as a Wan release's one
	// autoencoder file.
	release := filepath.Join(models, "release")
	vae := latentvideo.AutoencoderWeights(release)
	write(torchStateDict(t, "\x00\x00\xa0\x40"), "release", filepath.Base(vae))
	if other, err := Catalog[recipe.TaskReconstruction].Resolve(release); err != nil ||
		!slices.Equal(other.Inventory.Manifest.Components, []artifact.Component{component(artifact.ComponentWeights, 0, "vae/weights", vae)}) {
		t.Errorf("a directory holding one autoencoder file is served as %+v (%v), want that file alone", other.Inventory.Manifest.Components, err)
	}

	// A pair with a file gone is no model to train: the run would otherwise
	// record a model it could not have loaded.
	if err := os.Remove(sourceConfig); err != nil {
		t.Fatal(err)
	}
	if id, err := trainingworkflow.WeightsIdentity(bridge); err == nil {
		t.Errorf("a pair whose source encoder has no configuration was identified as %s", id)
	}
}
