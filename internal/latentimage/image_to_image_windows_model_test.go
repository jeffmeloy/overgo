//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/png"
	"math"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// Declared before the first run. ComfyUI encodes the picture with flux2-vae_small in bfloat16, overgo in F32, so the
// latents differ by bfloat16's rounding through the encoder; a picture scaled, read or normalized otherwise, or encoded
// with MOSS-Image-SFT's encoder and statistics, lies far outside both bounds.
const (
	sourceLatentCosine     = 0.999
	sourceLatentRelativeL2 = 0.04
)

// imageToImageFloor is the agreement in dB over squares the section's final latent keeps with ComfyUI's, the golden
// floor moss-ultra-compare/golden-floor sets for each section, set as it sets them: from the first measured run, less
// 3 dB. ComfyUI runs the Base's projections in int8 with its activations quantized per row, overgo the weights decoded
// to bfloat16, so the latents part over the 50 steps; from the picture mirrored, the same run kept -0.92 dB.
const imageToImageFloor = imageToImageMeasured - 3

// imageToImageMeasured is the agreement the first run of the section measured, 2026-11-17.
const imageToImageMeasured = 17.91

// The recording holds the section at denoise 0.6 (comfy-recorder's recorder, run denoise, -set 506.denoise=0.6, with
// the latents of the encode, 606:503, and of the sampler, 506), given the recorder's picture.
const (
	imageToImageSampler = "506"
	imageToImageEncode  = "606:503"
)

// TestImageToImageMeetsItsSectionFloor holds MOSS's image to image (moss-ultra-sections/img2img) to the MOSS Ultra
// workflow's section as ComfyUI ran it at denoise 0.6, on the workflow's own files (ultraModelRoot), with the
// section's prompts, guidance, sampler, schedule, seed and steps as its request and recording give them: the schedule
// cut by the denoise is ComfyUI's sigmas exactly; the picture, scaled and encoded, is ComfyUI's encoded latent within
// the declared bounds; and the final latent, sampled from σ₀ times the noise plus 1 − σ₀ times that latent, keeps the
// section's floor of agreement with ComfyUI's, where the same request from the picture mirrored does not. The section's
// positive prompt is empty, as the workflow saves it: the picture conditions it.
func TestImageToImageMeetsItsSectionFloor(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	recording := readComfySection(t, "denoise", "image-to-image")
	latent := func(node string) []float32 { return recording.latent(t, node) }
	picture := ultraPictureBytes(t)
	section := denoiseSection(t, recording)

	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	schedule := testcheck.Must(profile.rotarySchedule(ctx, section))(t)
	recorded := recording.Sigmas[imageToImageSampler]
	if len(schedule.Sigmas) != len(recorded) || schedule.Steps != section.Steps {
		t.Fatalf("the cut schedule has %d sigmas over %d steps; want ComfyUI's %d over %d", len(schedule.Sigmas), schedule.Steps, len(recorded), section.Steps)
	}
	for index, sigma := range recorded {
		if float32(schedule.Sigmas[index]) != float32(sigma) {
			t.Errorf("the cut schedule's sigma %d is %v; want ComfyUI's %v", index, schedule.Sigmas[index], sigma)
		}
	}

	sections := newPictureSections(root, section.Width, section.Height, nil)
	defer func() { testcheck.NoError(t, sections.close(context.WithoutCancel(ctx))) }()
	encoded := testcheck.Must(sections.sourceLatent(ctx, section.Image))(t)
	cosine, relative := latentAgreement(encoded, latent(imageToImageEncode))
	t.Logf("the encoded picture: cosine %.6f, relative L2 %.5f of ComfyUI's", cosine, relative)
	if !(cosine >= sourceLatentCosine) || !(relative <= sourceLatentRelativeL2) {
		t.Errorf("the encoded picture lies at cosine %.6f and relative L2 %.5f of ComfyUI's; want at least %g and at most %g", cosine, relative, sourceLatentCosine, sourceLatentRelativeL2)
	}

	want := latent(imageToImageSampler)
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, section))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	sampled := func(request Request) float64 {
		t.Helper()
		session := testcheck.Must(generator.prepare(ctx, request))(t)
		testcheck.Must(generator.integrate(ctx, session))(t)
		final := testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t)
		testcheck.Must(generator.decode(ctx, session))(t)
		return decibelsOverSquares(t, final.Data, want)
	}
	agreement := sampled(section)
	control := section
	control.Image = base64.StdEncoding.EncodeToString(mirrored(t, picture))
	testcheck.NoError(t, generator.Reset(ctx, control))
	other := sampled(control)
	t.Logf("the final latent keeps %.2f dB of agreement with ComfyUI's; from the picture mirrored, %.2f dB; the floor %.2f dB", agreement, other, imageToImageFloor)
	if !(agreement >= imageToImageFloor) {
		t.Errorf("the final latent keeps %.2f dB of agreement with ComfyUI's; want at least the section's floor, %.2f dB", agreement, imageToImageFloor)
	}
	if other >= imageToImageFloor {
		t.Errorf("from the picture mirrored the final latent keeps %.2f dB, the floor %.2f dB or more, so the floor does not hold the picture's part", other, imageToImageFloor)
	}
}

// mirrored is a PNG of picture's pixels mirrored left to right.
func mirrored(t *testing.T, picture []byte) []byte {
	t.Helper()
	decoded := testcheck.Must(png.Decode(bytes.NewReader(picture)))(t)
	bounds := decoded.Bounds()
	flipped := image.NewNRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			flipped.Set(bounds.Max.X-1-(x-bounds.Min.X), y, decoded.At(x, y))
		}
	}
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, flipped))
	return encoded.Bytes()
}

// latentAgreement is the cosine between two latents and the relative L2 distance of got from want.
func latentAgreement(got, want []float32) (cosine, relative float64) {
	if len(got) != len(want) || len(want) == 0 {
		return 0, math.Inf(1)
	}
	var dot, gotSquares, wantSquares, differenceSquares float64
	for index, value := range got {
		g, w := float64(value), float64(want[index])
		dot += g * w
		gotSquares += g * g
		wantSquares += w * w
		differenceSquares += (g - w) * (g - w)
	}
	return dot / math.Sqrt(gotSquares*wantSquares), math.Sqrt(differenceSquares / wantSquares)
}

// decibelsOverSquares is how far got agrees with want in dB: want's sum of squares over the sum of the differences'.
func decibelsOverSquares(t testing.TB, got, want []float32) float64 {
	t.Helper()
	// Two latents of different lengths, or a value that is not finite, are refused before they are scored (SQA-410): a
	// length apart had scored -Inf and a NaN sample NaN, each of which a control held only under its floor passed.
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("the latents hold %d and %d values; want one length", len(got), len(want))
	}
	var signal, noise float64
	for index, value := range got {
		w := float64(want[index])
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("the latents hold %g and %g at %d; want finite values", value, w, index)
		}
		signal += w * w
		noise += (float64(value) - w) * (float64(value) - w)
	}
	return 10 * math.Log10(signal/noise)
}

// denoiseSection is the image to image section's request as the denoise run's recording asks it: at a denoise below 1,
// 1024 x 576, from the recorder's picture.
func denoiseSection(t *testing.T, recording comfySection) Request {
	t.Helper()
	sampler := recording.inputs(imageToImageSampler)
	denoise, given := recording.Set[imageToImageSampler+".denoise"].(float64)
	if !given || denoise >= 1 || recording.Set["428.image"] != "overgo-recorder-input.png" || recording.Set["680.megapixels"] != 0.5625 ||
		sampler["sampler_name"] != "er_sde" || sampler["scheduler"] != sampling.ScheduleBongTangent || len(recording.Texts["585"]) != 1 || len(recording.Texts["586"]) != 1 {
		t.Fatalf("the recording set %v over a sampler of %v with prompts %v; want the section at a denoise below 1, 1024 x 576, from the recorder's picture", recording.Set, sampler, recording.Texts)
	}
	guidance := float32(sampler["cfg"].(float64))
	return Request{Prompt: recording.Texts["585"][0], NegativePrompt: recording.Texts["586"][0], GuidanceScale: &guidance, Width: 1024, Height: 576,
		Steps: int(sampler["steps"].(float64)), Seed: int64(sampler["seed"].(float64)), Denoise: denoise, Image: base64.StdEncoding.EncodeToString(ultraPictureBytes(t)),
		Schedule: &sampling.Schedule{Sampler: sampling.SamplerERSDE, Name: sampling.ScheduleBongTangent}}
}
