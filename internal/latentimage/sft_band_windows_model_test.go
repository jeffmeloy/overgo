//go:build modeltest && windows

package latentimage

import (
	"context"
	"encoding/base64"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	// The converter opens a Hugging Face directory as the model it serves (gguf.RegisterDirectoryOpener).
	_ "overgo/internal/hfconvert"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// The adapters the workflow's sections apply on SFT, by name, under a MOSS root's adapters/: the control adapter, and
// the Turbo sections' DMD2 (the SFT adapter is SFT itself).
const (
	sftControlAdapter = "moss-control"
	sftTurboAdapter   = "8-steps-dmd2-lora"
)

// sftModelRoot is MOSS-Image-SFT, the model overgo serves, with its official encoder, as a root that also holds the
// workflow's control and DMD2 adapters under adapters/ and, as its enhancer, the served Qwen3.5-9B
// (models/Qwen3.5-9B-Q8_0.gguf): each of SFT's files, each adapter and the enhancer hard-linked, nothing copied.
func sftModelRoot(t *testing.T) string {
	t.Helper()
	sft, ultra, root := testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f"), testutil.ModelArtifactDir(t, "moss-ultra"), t.TempDir()
	testcheck.NoError(t, filepath.WalkDir(sft, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		relative := testcheck.Must(filepath.Rel(sft, path))(t)
		if entry.IsDir() {
			return os.MkdirAll(filepath.Join(root, relative), 0o700)
		}
		return os.Link(path, filepath.Join(root, relative))
	}))
	testcheck.NoError(t, os.Mkdir(filepath.Join(root, "adapters"), 0o700))
	for _, name := range []string{sftControlAdapter, sftTurboAdapter} {
		testcheck.NoError(t, os.Link(filepath.Join(ultra, "adapters", name+".safetensors"), filepath.Join(root, "adapters", name+".safetensors")))
	}
	testcheck.NoError(t, os.Link(testutil.ModelArtifactDir(t, "Qwen3.5-9B-Q8_0.gguf"), filepath.Join(root, enhancerDirectory)))
	return root
}

// sftTurboCases are the workflow's Turbo sections whose adapters on SFT are DMD2 alone, each recorded at 1024 x 576: the
// depth Turbo section, moss-control and DMD2 together, waits for moss-eight-step.
var sftTurboCases = []goldenCase{
	{name: "text to image, Turbo", run: "pass-1", slug: "text-to-image-turbo", sampler: "331", finished: "147",
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			request := textSection(t, recording, "331", "27", "325", "336")
			request.Adapter = sftTurboAdapter
			return request
		}},
	{name: "image to image, Turbo", run: "pass-1", slug: "image-to-image-turbo", sampler: "198", finished: "344",
		request: func(t *testing.T, recording comfySection) Request {
			t.Helper()
			if recording.Set["8.image"] != "overgo-recorder-input.png" {
				t.Fatalf("the recording set %v; want the recorder's picture", recording.Set)
			}
			request := textSection(t, recording, "198", "316", "317", "343")
			request.Adapter, request.Denoise = sftTurboAdapter, recording.inputs("198")["denoise"].(float64)
			request.Image = ultraPicture64(t)
			return request
		}},
}

// TestServedMossCompletesEveryUltraSection runs the workflow's sections on MOSS-Image-SFT, the model overgo serves
// (moss-ultra-compare/sft-band): each section and prompt mode the golden floors hold on the workflow's own files, and
// the Turbo sections whose adapters on SFT are DMD2 alone, asked as their recordings ask them, mode 2 enhanced from the
// recording's source and exclusions by the served Qwen3.5-9B. SFT is not the Base the golden was drawn on, so each
// section's agreement with ComfyUI's, its final latent's and its finished picture's in dB over squares, is recorded as
// the band, its lowest and highest; the check holds that every section completes with a picture.
func TestServedMossCompletesEveryUltraSection(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := sftModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	testcheck.Must(modelrecipe.PublishArchitectureProfileCatalog(ctx, store))(t)
	band := struct{ latentLow, latentHigh, pictureLow, pictureHigh float64 }{math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)}
	for _, section := range append(append([]goldenCase(nil), goldenCases...), sftTurboCases...) {
		recording := readComfySection(t, section.run, section.slug)
		request := section.request(t, recording)
		if section.name == "prompt mode 2" {
			// SFT's mode 2 enhances the recording's own source and exclusions, as the workflow's enhancer did.
			request.Prompt, request.NegativePrompt, request.Enhance = field(recording.inputs("459"), "value"), field(recording.inputs("460"), "value"), true
			request = testcheck.Must(EnhanceRequest(ctx, enhancerOpener(store), root, request))(t)
		}
		generator := testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
		session := testcheck.Must(generator.prepare(ctx, request))(t)
		testcheck.Must(generator.integrate(ctx, session))(t)
		final := testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t)
		picture := testcheck.Must(generator.decode(ctx, session))(t)
		closeReporting(t, context.WithoutCancel(ctx), generator)
		if len(picture.Data) == 0 {
			t.Fatalf("%s drew no picture on SFT", section.name)
		}
		latent, finished := decibelsOverSquares(t, final.Data, recording.latent(t, section.sampler)), pictureDecibels(t, picture.Data, recording.picture(t, section.finished))
		band.latentLow, band.latentHigh = min(band.latentLow, latent), max(band.latentHigh, latent)
		band.pictureLow, band.pictureHigh = min(band.pictureLow, finished), max(band.pictureHigh, finished)
		t.Logf("%s on SFT: the final latent keeps %.2f dB of the golden's, the finished picture %.2f dB over %d-pixel squares", section.name, latent, finished, pictureSquare)
	}
	t.Logf("SFT's band against the golden: final latents %.2f to %.2f dB, finished pictures %.2f to %.2f dB", band.latentLow, band.latentHigh, band.pictureLow, band.pictureHigh)
}

// ultraPicture64 is the recorder's picture as a request carries it, in base64.
func ultraPicture64(t *testing.T) string {
	t.Helper()
	return base64.StdEncoding.EncodeToString(ultraPictureBytes(t))
}

// field is a request node's input of name as text, empty where it holds none.
func field(inputs map[string]any, name string) string {
	value, _ := inputs[name].(string)
	return value
}
