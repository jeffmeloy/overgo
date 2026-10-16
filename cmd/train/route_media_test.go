package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestTrainRoutesReachTheMediaTrainers publishes diffusion training through the command route.
func TestTrainRoutesReachTheMediaTrainers(t *testing.T) {
	t.Run("diffusionimage", func(t *testing.T) {
		root := t.TempDir()
		model := writeTinyDiffusionImage(t, filepath.Join(root, "model"))
		picture := image.NewRGBA(image.Rect(0, 0, tinyImageSide, tinyImageSide))
		for x := range tinyImageSide {
			for y := range tinyImageSide {
				picture.Set(x, y, color.RGBA{R: uint8(x * tinyImageStride), G: uint8(y * tinyImageStride), B: uint8((x + y) * tinyImageStride), A: math.MaxUint8})
			}
		}
		file, err := os.Create(filepath.Join(root, "gradient.png"))
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(file, picture); err != nil {
			t.Fatal(err)
		}
		if err := file.Close(); err != nil {
			t.Fatal(err)
		}
		manifest := filepath.Join(root, "manifest.txt")
		if err := os.WriteFile(manifest, []byte("gradient.png\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		requireRoutePublishes(t, "diffusionimage", "-model", model, "-images", manifest)
	})

}

// routeTestSteps is how many updates a route test trains.
const routeTestSteps = 2

// requireRoutePublishes trains route with args for routeTestSteps updates
// into a fresh store and holds it to publishing a checkpoint recorded there
// at its last step, which it returns.
func requireRoutePublishes(t *testing.T, route string, args ...string) (trainingprogram.Checkpoint, string) {
	t.Helper()
	root := t.TempDir()
	storePath, out := filepath.Join(root, "store"), filepath.Join(root, "checkpoint")
	args = append([]string{route, "-steps", fmt.Sprint(routeTestSteps), "-store", storePath, "-out", out}, args...)
	if err := runRoute(t.Context(), args); err != nil {
		t.Fatalf("%s route: %v", route, err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(out)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, recorded, err := store.Artifact(t.Context(), checkpoint.ID()); err != nil || !recorded || checkpoint.Optimizer.Step != routeTestSteps {
		t.Fatalf("checkpoint %s at step %d recorded %t (%v), want the route's %d steps recorded", checkpoint.ID(), checkpoint.Optimizer.Step, recorded, err, routeTestSteps)
	}
	return checkpoint, out
}

// The generated training image is an 8x8 colour gradient.
const tinyImageSide, tinyImageStride = 8, 32

// writeTinyDiffusionImage writes the committed tiny u-vit golden model, its
// weights generated as the golden's generator does, as a checkpoint
// directory the diffusionimage loader reads, and returns the directory.
func writeTinyDiffusionImage(t *testing.T, directory string) string {
	t.Helper()
	var fixture struct {
		WeightKeys   []string
		WeightShapes map[string][]int
	}
	if err := jsonfile.Decode(testutil.FixturePath(t, "simplediffusion", "udit_tiny.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	// The golden generator's pattern: value i of key k is
	// ((i+(k+1)*7)%29-14)/128.
	const offsetStride, period, center, scale = 7, 29, 14, 128
	tensors := make(map[string][]float32, len(fixture.WeightKeys))
	for index, key := range fixture.WeightKeys {
		values := make([]float32, elementCount(fixture.WeightShapes[key]))
		for element := range values {
			values[element] = float32((element+(index+1)*offsetStride)%period-center) / scale
		}
		tensors[key] = values
	}
	writeModelDirectory(t, directory, "config.json", `{"model_type":"uvit"}`, tensors, fixture.WeightShapes)
	return directory
}

// writeModelDirectory writes a model directory: its declaration and its
// float32 weights.
func writeModelDirectory(t *testing.T, directory, declaration, content string, tensors map[string][]float32, shapes map[string][]int) {
	t.Helper()
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, declaration), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), tensors, shapes, nil); err != nil {
		t.Fatal(err)
	}
}

// elementCount is how many values a tensor of shape holds.
func elementCount(shape []int) int {
	count := 1
	for _, extent := range shape {
		count *= extent
	}
	return count
}
