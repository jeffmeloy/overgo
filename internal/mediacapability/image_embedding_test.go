//go:build modeltest

package mediacapability

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

// TestImageEmbeddingCapability runs the catalog's image embedding end to end
// on the registered webssl-dino checkpoint: the recipe resolves and compiles,
// an encoded image returns one value per hidden channel,
// the same image embeds identically twice, a different image embeds
// differently, and a checkpoint whose family no reviewed declaration answers
// for is refused before any tensor is bound.
func TestImageEmbeddingCapability(t *testing.T) {
	t.Parallel()
	directory := modeltest.Directory(t, "webssl-dino300m-full2b-224")
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	capability := Catalog[recipe.TaskImageEmbedding]
	embed := func(t *testing.T, directory string, encoded []byte) ([]float32, error) {
		t.Helper()
		source, err := capability.Resolve(t.Context(), directory)
		testcheck.NoError(t, err)
		definition, err := modelrecipe.CapabilityDefinition(recipe.TaskImageEmbedding, source.Inventory.Manifest.ID)
		testcheck.NoError(t, err)
		program, err := modelrecipe.CompileCapability(definition)
		testcheck.NoError(t, err)
		request, err := json.Marshal(map[string]any{"image": encoded})
		testcheck.NoError(t, err)
		output, err := capability.Execute(t.Context(), store, directory, modelrecipe.CapabilityEvidenceSelection{Program: program}, string(request))
		if err != nil {
			return nil, err
		}
		embedding, ok := capabilityruntime.Unwrap(output).([]float32)
		if !ok {
			t.Fatalf("image embedding output is %T", capabilityruntime.Unwrap(output))
		}
		return embedding, nil
	}
	gradient := encodePNG(t, 240, 300, func(x, y, c int) uint8 { return uint8((x*7 + y*13 + c*29 + (x*y)%17) % 256) })
	checker := encodePNG(t, 240, 300, func(x, y, c int) uint8 { return uint8(255 * ((x/20 + y/20 + c) % 2)) })
	first, err := embed(t, directory, gradient)
	testcheck.NoError(t, err)
	var config struct {
		Hidden int `json:"hidden_size"`
	}
	data, err := os.ReadFile(filepath.Join(directory, "config.json"))
	testcheck.NoError(t, err)
	testcheck.NoError(t, json.Unmarshal(data, &config))
	if len(first) != config.Hidden {
		t.Fatalf("embedding has %d values, the checkpoint's hidden size is %d", len(first), config.Hidden)
	}
	again, err := embed(t, directory, gradient)
	testcheck.NoError(t, err)
	if !equalEmbeddings(first, again) {
		t.Fatal("the same image embedded differently twice")
	}
	other, err := embed(t, directory, checker)
	testcheck.NoError(t, err)
	if cosine(first, other) > 0.99 {
		t.Fatalf("different images embed alike: cosine %.4f", cosine(first, other))
	}
	// A copy of the checkpoint's configuration naming another family is
	// refused before its weights are read.
	foreign := t.TempDir()
	for _, name := range []string{"config.json", "preprocessor_config.json", "model.safetensors"} {
		source := filepath.Join(directory, name)
		if name == "config.json" {
			testcheck.NoError(t, os.WriteFile(filepath.Join(foreign, name), bytes.Replace(data, []byte(`"dinov2"`), []byte(`"unreviewed"`), 1), 0o600))
			continue
		}
		if err := os.Link(source, filepath.Join(foreign, name)); err != nil {
			t.Skip("hard links are unavailable here: " + err.Error())
		}
	}
	if _, err := embed(t, foreign, gradient); err == nil || !strings.Contains(err.Error(), "no reviewed declaration") {
		t.Fatalf("a checkpoint without a reviewed declaration was bound: %v", err)
	}
}

func encodePNG(t *testing.T, height, width int, value func(x, y, c int) uint8) []byte {
	t.Helper()
	picture := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			offset := picture.PixOffset(x, y)
			for c := range 3 {
				picture.Pix[offset+c] = value(x, y, c)
			}
			picture.Pix[offset+3] = math.MaxUint8
		}
	}
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, picture))
	return encoded.Bytes()
}

func equalEmbeddings(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}

func cosine(a, b []float32) float64 {
	var dot, aNorm, bNorm float64
	for index := range a {
		dot += float64(a[index]) * float64(b[index])
		aNorm += float64(a[index]) * float64(a[index])
		bNorm += float64(b[index]) * float64(b[index])
	}
	return dot / math.Sqrt(aNorm*bNorm)
}
