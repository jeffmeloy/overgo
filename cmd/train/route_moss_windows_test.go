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
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
)

// mossRouteFixture writes a compact checkpoint's manifest of gradient pictures
// and captions, and returns the checkpoint and the manifest.
func mossRouteFixture(t *testing.T) (root, manifest string) {
	t.Helper()
	root = testutil.ExtractTarGzip(t, filepath.Join("..", "..", "fixtures", "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	// The checkpoint's layout: its weights are all in the components model_index.json names.
	testcheck.NoError(t, os.Remove(filepath.Join(root, "reference.safetensors")))
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
		testcheck.NoError(t, png.Encode(&encoded, picture))
		name := fmt.Sprintf("picture-%d.png", index)
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), encoded.Bytes(), 0o600))
		fmt.Fprintf(&lines, "%s\t%s\n", name, caption)
	}
	manifest = filepath.Join(directory, "pictures.tsv")
	testcheck.NoError(t, os.WriteFile(manifest, lines.Bytes(), 0o600))
	return root, manifest
}

// TestInterleavedRowsKeepTheProcessorsRange holds a picture to the signed unit
// range the encoder reads, which is the processor's own. Stretching it again
// makes the pictures darker and harsher, and training then learns exactly that
// while its loss falls.
func TestInterleavedRowsKeepTheProcessorsRange(t *testing.T) {
	t.Parallel()
	picture := image.NewRGBA(image.Rect(0, 0, 2, 2))
	picture.SetRGBA(0, 0, color.RGBA{255, 255, 255, 255})
	picture.SetRGBA(1, 1, color.RGBA{0, 0, 0, 255})
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, picture))
	example := testcheck.Must(trainingdata.ImageProcessor(trainingdata.RoleTarget)(t.Context(), trainingdata.RawRecord{ID: "picture", Group: "picture", Data: encoded.Bytes()}))(t)
	rows := testcheck.Must(interleavedRows(example.Values[0], 2))(t)
	const channels = 3
	for channel := range channels {
		if white, black := rows[channel], rows[len(rows)-channels+channel]; white != 1 || black != -1 {
			t.Fatalf("channel %d: white %v, black %v, want 1 and -1", channel, white, black)
		}
	}
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
	store, out, adapterOut, served := t.TempDir(), filepath.Join(scratch, "checkpoint"), filepath.Join(scratch, "adapter"), filepath.Join(scratch, "served")
	arguments := []string{"moss", "-model", root, "-pictures", manifest, "-size", fmt.Sprint(mossRouteSize), "-rank", "2", "-alpha", "2", "-steps", "6", "-store", store, "-out", out, "-adapter", adapterOut, "-served", served}
	heldout, report := filepath.Join(filepath.Dir(manifest), "heldout.tsv"), filepath.Join(scratch, "heldout.json")
	testcheck.NoError(t, os.WriteFile(heldout, []byte("held-0.png\ta red square\n"), 0o600))
	picture := testcheck.Must(os.ReadFile(filepath.Join(filepath.Dir(manifest), "picture-0.png")))(t)
	// A byte past its end leaves the pixels the reading is pinned to and makes it another file: a copy is refused.
	testcheck.NoError(t, os.WriteFile(filepath.Join(filepath.Dir(manifest), "held-0.png"), append(picture, 0), 0o600))
	arguments = append(arguments, "-heldout", heldout, "-report", report)
	testcheck.NoError(t, runRoute(t.Context(), arguments))
	adapter := testcheck.Must(peft.Load(adapterOut))(t)
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
	base := testcheck.Must(safetensors.OpenSource(filepath.Join(root, "transformer")))(t)
	defer base.Close()
	trained := testcheck.Must(safetensors.OpenSource(filepath.Join(served, "transformer")))(t)
	defer trained.Close()
	changed, kept := 0, 0
	for name, tensor := range base.Tensors {
		before := testcheck.Must(safetensors.ReadF32(tensor))(t)
		after := testcheck.Must(safetensors.ReadF32(trained.Tensors[name]))(t)
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
	testcheck.NoError(t, jsonfile.Decode(report, &reading))
	if len(reading.Base) != 1 || len(reading.Trained) != 1 || !(reading.Base[0] > 0) || reading.Base[0] == reading.Trained[0] {
		t.Fatalf("held-out reading %+v", reading)
	}
	t.Run("cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		out := filepath.Join(t.TempDir(), "checkpoint")
		cancelled := append([]string{}, arguments[:len(arguments)-10]...)
		cancelled = append(cancelled, "-store", t.TempDir(), "-out", out)
		if err := runRoute(ctx, cancelled); !errors.Is(err, context.Canceled) {
			t.Fatal("cancellation lost", err)
		}
		if _, err := os.Stat(out); err == nil {
			t.Fatal("a cancelled run published")
		}
	})
}
