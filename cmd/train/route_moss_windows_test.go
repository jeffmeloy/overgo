//go:build windows

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/latentimage"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// mossRouteFixture writes a compact checkpoint's manifest of gradient pictures
// and captions, and returns the checkpoint and the manifest.
func mossRouteFixture(t *testing.T) (root, manifest string) {
	t.Helper()
	root = testutil.ExtractTarGzip(t, filepath.Join("..", "..", "fixtures", "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	// The checkpoint's layout: its weights are all in the components model_index.json names.
	if err := os.Remove(filepath.Join(root, "reference.safetensors")); err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	var lines bytes.Buffer
	for index, caption := range []string{"a red square", "a green square", "a blue square"} {
		picture := image.NewRGBA(image.Rect(0, 0, mossRouteSize, mossRouteSize))
		for y := range mossRouteSize {
			for x := range mossRouteSize {
				shade := [3]uint8{uint8(4 * x), uint8(4 * y), 32}
				shade[index] = 240
				picture.SetRGBA(x, y, color.RGBA{shade[0], shade[1], shade[2], 255})
			}
		}
		var encoded bytes.Buffer
		if err := png.Encode(&encoded, picture); err != nil {
			t.Fatal(err)
		}
		name := fmt.Sprintf("picture-%d.png", index)
		if err := os.WriteFile(filepath.Join(directory, name), encoded.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&lines, "%s\t%s\n", name, caption)
	}
	manifest = filepath.Join(directory, "pictures.tsv")
	if err := os.WriteFile(manifest, lines.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return root, manifest
}

// mossRouteSize is a compact checkpoint's smallest picture.
const mossRouteSize = 32

// TestMOSSRoute trains a compact checkpoint's adapter from captioned pictures
// through the whole route: the frozen path once, steps that descend, a
// checkpoint, a PEFT adapter whose up factors moved, and a served directory
// whose transformer differs from its base and whose other components are the
// base's. A cancelled run publishes nothing and releases the device.
func TestMOSSRoute(t *testing.T) {
	cudatest.Require(t)
	root, manifest := mossRouteFixture(t)
	scratch := t.TempDir()
	store, out, adapterOut, served := filepath.Join(scratch, "store"), filepath.Join(scratch, "checkpoint"), filepath.Join(scratch, "adapter"), filepath.Join(scratch, "served")
	arguments := []string{"moss", "-model", root, "-pictures", manifest, "-size", fmt.Sprint(mossRouteSize), "-rank", "2", "-alpha", "2", "-steps", "6", "-store", store, "-out", out, "-adapter", adapterOut, "-served", served}
	heldout, report := filepath.Join(filepath.Dir(manifest), "heldout.tsv"), filepath.Join(scratch, "heldout.json")
	if err := os.WriteFile(heldout, []byte("held-0.png\ta red square\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	picture, err := os.ReadFile(filepath.Join(filepath.Dir(manifest), "picture-0.png"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(manifest), "held-0.png"), picture, 0o600); err != nil {
		t.Fatal(err)
	}
	arguments = append(arguments, "-heldout", heldout, "-report", report)
	if err := runRoute(t.Context(), arguments); err != nil {
		t.Fatal(err)
	}
	adapter, err := peft.Load(adapterOut)
	if err != nil {
		t.Fatal(err)
	}
	if len(adapter.Modules) == 0 {
		t.Fatal("the adapter holds no modules")
	}
	moved := false
	for _, module := range adapter.Modules {
		for _, value := range module.Up {
			moved = moved || value != 0
		}
	}
	if !moved {
		t.Fatal("training moved no up factor")
	}
	base, err := safetensors.OpenSource(filepath.Join(root, "transformer"))
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	trained, err := safetensors.OpenSource(filepath.Join(served, "transformer"))
	if err != nil {
		t.Fatal(err)
	}
	defer trained.Close()
	changed, kept := 0, 0
	for name, tensor := range base.Tensors {
		before, err := safetensors.ReadF32(tensor)
		if err != nil {
			t.Fatal(err)
		}
		after, err := safetensors.ReadF32(trained.Tensors[name])
		if err != nil {
			t.Fatal(err)
		}
		same := len(before) == len(after)
		for index := 0; same && index < len(before); index++ {
			same = before[index] == after[index]
		}
		if same {
			kept++
		} else {
			changed++
		}
	}
	if changed == 0 || changed > len(adapter.Modules) || kept == 0 {
		t.Fatalf("%d tensors changed and %d kept, with %d modules", changed, kept, len(adapter.Modules))
	}
	if _, err := latentimage.ResolveProfile(served); err != nil {
		t.Fatal("the served directory does not resolve as the base does:", err)
	}
	if _, err := os.Stat(filepath.Join(out, trainingprogram.CheckpointWeights)); err != nil {
		t.Fatal("the checkpoint holds no factors:", err)
	}
	var reading mossHeldout
	if err := jsonfile.Decode(report, &reading); err != nil {
		t.Fatal(err)
	}
	if len(reading.Base) != 1 || len(reading.Trained) != 1 || !(reading.Base[0] > 0) || reading.Base[0] == reading.Trained[0] {
		t.Fatalf("held-out reading %+v", reading)
	}
	t.Run("cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		out := filepath.Join(t.TempDir(), "checkpoint")
		cancelled := append([]string{}, arguments[:len(arguments)-10]...)
		cancelled = append(cancelled, "-store", filepath.Join(t.TempDir(), "store"), "-out", out)
		if err := runRoute(ctx, cancelled); !errors.Is(err, context.Canceled) {
			t.Fatal("cancellation lost", err)
		}
		if _, err := os.Stat(out); err == nil {
			t.Fatal("a cancelled run published")
		}
	})
}
