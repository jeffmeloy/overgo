//go:build windows && modeltest

package main

import (
	"bytes"
	"image"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/vitencoder"
)

// screenFixtureClips is how many VidGen source clips the screen's fixture is cut from: a case each.
const screenFixtureClips = 32

// TestTheScreenFailsACandidateThatLostItsPrompt holds the media screen to what it is for. Its fixture is cut from VidGen
// source clips, each clip's caption a prompt: the reference path's output is a clip's first sampled frame, and the
// reference path's output at another seed is a frame of the same clip seconds later, another picture the caption
// allows. A candidate that rounds differently, each channel one level off, keeps every case and passes; a candidate
// whose outputs lost their prompt, each the next clip's first frame, lies farther from the reference than the other
// seed and fails. Both are embedded with webssl-dino300m on the card.
func TestTheScreenFailsACandidateThatLostItsPrompt(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	ffmpeg := testcheck.Must(media.FFmpeg())(t)
	clips := filepath.Join(roots.Datasets, "VidGen")
	listed := testcheck.Must(os.ReadDir(filepath.Join(roots.Datasets, "VidGen_frames")))(t)
	var first, later [][]byte
	var prompts []string
	for _, entry := range listed {
		clip := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		frames, err := media.DecodeVideoFile(ctx, filepath.Join(clips, clip+clipFileExtension), ffmpeg, frameSampleFPS, frameSampleCount)
		if err != nil || len(frames) < frameHalf {
			continue
		}
		first, later = append(first, pngBytes(t, frames[0])), append(later, pngBytes(t, frames[len(frames)-1]))
		prompts = append(prompts, clip)
		if len(prompts) == screenFixtureClips {
			break
		}
	}
	if len(prompts) < screenFixtureClips {
		t.Fatalf("VidGen holds %d clips of two sampled frames or more among VidGen_frames' names, not %d", len(prompts), screenFixtureClips)
	}
	rounded := make([][]byte, len(first))
	for index, picture := range first {
		rounded[index] = pngBytes(t, offByOne(t, picture))
	}
	lost := append(slices.Clone(first[1:]), first[0])
	referenceSeed, otherSeed := "1", "2"
	reference := writeScreenDirectory(t, prompts, first, referenceSeed)
	reseeded := writeScreenDirectory(t, prompts, later, otherSeed)
	encoder := testcheck.Must(vitencoder.Load(roots.ResolveModelPath("webssl-dino300m-full2b-224")))(t)
	card := testcheck.Must(encoder.OpenCard(ctx, 0))(t)
	t.Cleanup(func() { testcheck.NoError(t, card.Close()) })
	for _, candidate := range []struct {
		name    string
		outputs [][]byte
		passes  bool
	}{{"rounds differently", rounded, true}, {"lost its prompt", lost, false}} {
		screen := testcheck.Must(screenPaths(ctx, card, writeScreenDirectory(t, prompts, candidate.outputs, referenceSeed), reference, reseeded))(t)
		t.Logf("a candidate that %s lost %d of %d cases, kept %d, tied %d: chance %.3g at level %.3g, passed %t",
			candidate.name, screen.Lost, len(screen.Cases), screen.Kept, screen.Ties, screen.Tail, screen.Level, screen.Passed)
		if screen.Passed != candidate.passes {
			t.Errorf("a candidate that %s passed the screen: %t, want %t", candidate.name, screen.Passed, candidate.passes)
		}
	}
}

// writeScreenDirectory writes outputs as a render directory: each a picture with the manifest's draw of its prompt at
// seed.
func writeScreenDirectory(t *testing.T, prompts []string, outputs [][]byte, seed string) string {
	t.Helper()
	directory := t.TempDir()
	draws := map[string]any{}
	for index, output := range outputs {
		file := prompts[index] + ".png"
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, file), output, frameFileMode))
		draws[prompts[index]] = map[string]any{"file": file, "request": map[string]any{"prompt": prompts[index], "seed": seed}}
	}
	testcheck.NoError(t, jsonfile.Write(filepath.Join(directory, renderManifestFile), map[string]any{"draws": draws}, frameFileMode))
	return directory
}

// offByOne is picture with each channel one level up or down, by the parity of its place: an output that rounds
// differently.
func offByOne(t *testing.T, picture []byte) image.Image {
	t.Helper()
	decoded := testcheck.Must(png.Decode(bytes.NewReader(picture)))(t)
	shifted := image.NewNRGBA(decoded.Bounds())
	draw.Draw(shifted, shifted.Bounds(), decoded, decoded.Bounds().Min, draw.Src)
	// The pixels are laid red, green, blue, alpha; alpha is left as it is.
	const channels, alpha = 4, 3
	for index := range shifted.Pix {
		if index%channels == alpha {
			continue
		}
		switch level := shifted.Pix[index]; {
		case index%2 == 0 && level < math.MaxUint8:
			shifted.Pix[index]++
		case level > 0:
			shifted.Pix[index]--
		}
	}
	return shifted
}

func pngBytes(t *testing.T, picture image.Image) []byte {
	t.Helper()
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, picture))
	return encoded.Bytes()
}
