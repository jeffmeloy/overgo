package main

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestMOSSRouteRefusesAHeldOutCopyOfATrainedPicture: a held-out picture whose model input is a trained one's is refused,
// naming both: a copy under another name, the file with one byte appended past its end, and the same pixels encoded
// again. By the files' bytes the last two had passed as held out, the route's own test holding the padded copy as its
// held-out picture (SQA-506). Another picture passes, and so does a manifest of one; a picture that cannot be read or
// decoded is refused.
func TestMOSSRouteRefusesAHeldOutCopyOfATrainedPicture(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	write := func(name string, data []byte) mossPicture {
		t.Helper()
		path := filepath.Join(directory, name)
		testcheck.NoError(t, os.WriteFile(path, data, 0o600))
		return mossPicture{path: path, caption: name}
	}
	encode := func(level png.CompressionLevel, red uint8) []byte {
		const side = 4
		picture := image.NewRGBA(image.Rect(0, 0, side, side))
		for y := range side {
			for x := range side {
				picture.SetRGBA(x, y, color.RGBA{red, uint8(16 * x), uint8(16 * y), 255})
			}
		}
		var encoded bytes.Buffer
		testcheck.NoError(t, (&png.Encoder{CompressionLevel: level}).Encode(&encoded, picture))
		return encoded.Bytes()
	}
	owl := encode(png.DefaultCompression, 200)
	trained := []mossPicture{write("fox.png", encode(png.DefaultCompression, 40)), write("owl.png", owl)}
	other := write("cat.png", encode(png.DefaultCompression, 120))
	copies := map[string][]byte{"renamed.png": owl, "padded.png": append(slices.Clone(owl), 0), "reencoded.png": encode(png.NoCompression, 200)}
	for name, data := range copies {
		err := refuseTrainedHeldOut(trained, []mossPicture{other, write(name, data)})
		if err == nil || !strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), "owl.png") {
			t.Fatalf("a held-out %s = %v, want it refused, naming both", name, err)
		}
	}
	if err := refuseTrainedHeldOut(trained, []mossPicture{other}); err != nil {
		t.Fatalf("another picture = %v", err)
	}
	for _, unread := range []mossPicture{{path: filepath.Join(directory, "absent.png")}, write("text.png", []byte("no picture"))} {
		if err := refuseTrainedHeldOut(trained, []mossPicture{unread}); err == nil {
			t.Fatalf("a held-out picture that cannot be read or decoded, %s, passes", unread.path)
		}
	}
}

// TestAMOSSRunIsRefusedBeforeItsFirstPicture: what the files alone refuse is refused before a picture is prepared on the
// device: a held-out picture that is a trained one with a byte appended, and a training grid shifted from the model's.
func TestAMOSSRunIsRefusedBeforeItsFirstPicture(t *testing.T) {
	t.Parallel()
	root, manifest := mossRouteFixture(t)
	directory := filepath.Dir(manifest)
	write := func(path, content string) {
		testutil.WriteTextFile(t, filepath.Dir(path), filepath.Base(path), content, 0o600)
	}
	trained := testcheck.Must(os.ReadFile(filepath.Join(directory, "picture-0.png")))(t)
	write(filepath.Join(directory, "held-0.png"), string(append(trained, 0)))
	held := filepath.Join(directory, "heldout.tsv")
	write(held, "held-0.png\ta red square\n")
	route := func(extra ...string) error {
		scratch := t.TempDir()
		return runRoute(t.Context(), append([]string{
			"moss", "-model", root, "-pictures", manifest, "-size", "32", "-rank", "2", "-alpha", "2", "-steps", "1",
			"-store", t.TempDir(), "-out", filepath.Join(scratch, "checkpoint"),
		}, extra...))
	}
	report := filepath.Join(t.TempDir(), "heldout.json")
	if err := route("-heldout", held, "-report", report); err == nil || !strings.Contains(err.Error(), "is held out and holds the picture of") {
		t.Fatalf("a held-out copy = %v, want it refused before preparation", err)
	}
	scheduler := filepath.Join(root, "scheduler", "scheduler_config.json")
	raw := testcheck.Must(os.ReadFile(scheduler))(t)
	var config map[string]any
	testcheck.NoError(t, json.Unmarshal(raw, &config))
	config["training_shift"] = 3
	shifted := testcheck.Must(json.Marshal(config))(t)
	write(scheduler, string(shifted))
	if err := route(); err == nil || !strings.Contains(err.Error(), "training_shift") {
		t.Fatalf("a shifted training grid = %v, want it refused before preparation", err)
	}
}
