//go:build modeltest && windows

package latentimage

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/png"
	"path/filepath"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/sampling"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/vitencoder"
)

// depthControlFloor is the agreement in dB over squares the depth section's final latent keeps with ComfyUI's, set as
// moss-ultra-compare/golden-floor sets each section's: from the first measured run, less 3 dB. The section samples its
// whole schedule from noise, where image to image's starts 0.6 of the way, so ComfyUI's int8 activations against
// overgo's bfloat16 weights part the latents over more of it; from the map mirrored, the same run kept -1.04 dB.
const depthControlFloor = depthControlMeasured - 3

// depthControlMeasured is the agreement the first run of the section measured, 2026-11-17.
const depthControlMeasured = 14.13

// The depth section's nodes (fixtures/comfy/moss-ultra/requests/depth-control.json): its sampler, its control
// (MOSSImageControl) and its depth estimator (DepthAnythingV2Preprocessor).
const (
	depthControlSampler   = "443"
	depthControlNode      = "612:441"
	depthControlEstimator = "612:438"
)

// TestDepthControlMeetsItsSectionFloor holds MOSS's depth control (moss-ultra-sections/depth-control) to the MOSS Ultra
// workflow's depth section as ComfyUI ran it (comfy-recorder/record-rest, pass-1), on the workflow's own files with its
// control adapter, moss-control: the recorder's picture on the canvas, its depth map as overgo's Depth Anything V2
// makes it at the section's resolution, scaled back to the canvas, encoded and read beside the latent through the
// adapter's widened embedder for the section's share of the sampling, with its prompts, guidance, sampler, schedule,
// seed and steps and no blocks skipped, as the section skips none. The schedule is ComfyUI's sigmas exactly; the final
// latent keeps the section's floor of agreement with ComfyUI's, where the same request from the map mirrored does not.
func TestDepthControlMeetsItsSectionFloor(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	recording := readComfySection(t, "pass-1", "depth-control")
	section, depthMap := depthControlSection(t, recording)
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	schedule := testcheck.Must(profile.rotarySchedule(ctx, section))(t)
	recorded := recording.Sigmas[depthControlSampler]
	if len(schedule.Sigmas) != len(recorded) {
		t.Fatalf("the schedule has %d sigmas; want ComfyUI's %d", len(schedule.Sigmas), len(recorded))
	}
	for index, sigma := range recorded {
		if float32(schedule.Sigmas[index]) != float32(sigma) {
			t.Errorf("the schedule's sigma %d is %v; want ComfyUI's %v", index, schedule.Sigmas[index], sigma)
		}
	}
	low, high := section.Control.window(profile.Rotary.Shift)
	t.Logf("the control runs from sigma %.6f to %.6f", high, low)

	want := recording.latent(t, depthControlSampler)
	generator := testcheck.Must(LoadGenerator(ctx, root, profile, section))(t)
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	sampled := func(request Request) float64 {
		t.Helper()
		session := testcheck.Must(generator.prepare(ctx, request))(t)
		testcheck.Must(generator.integrate(ctx, session))(t)
		final := testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t)
		testcheck.Must(generator.decode(ctx, session))(t)
		return decibelsOverSquares(final.Data, want)
	}
	agreement := sampled(section)
	other := section
	mirroredControl := *section.Control
	mirroredControl.Map = base64.StdEncoding.EncodeToString(mirrored(t, depthMap))
	other.Control = &mirroredControl
	testcheck.NoError(t, generator.Reset(ctx, other))
	from := sampled(other)
	t.Logf("the final latent keeps %.2f dB of agreement with ComfyUI's; from the map mirrored, %.2f dB; the floor %.2f dB", agreement, from, depthControlFloor)
	if !(agreement >= depthControlFloor) {
		t.Errorf("the final latent keeps %.2f dB of agreement with ComfyUI's; want at least the section's floor, %.2f dB", agreement, depthControlFloor)
	}
	if from >= depthControlFloor {
		t.Errorf("from the map mirrored the final latent keeps %.2f dB, the floor %.2f dB or more, so the floor does not hold the map's part", from, depthControlFloor)
	}
}

// depthControlSection is the depth section's request as its recording asks it, at 1024 x 576 from the recorder's
// picture, and the depth map it reads, a PNG of overgo's Depth Anything V2 map at the section's resolution.
func depthControlSection(t *testing.T, recording comfySection) (Request, []byte) {
	t.Helper()
	sampler, control, estimator := recording.inputs(depthControlSampler), recording.inputs(depthControlNode), recording.inputs(depthControlEstimator)
	if recording.Set["513.image"] != "overgo-recorder-input.png" || recording.Set["690.megapixels"] != 0.5625 || sampler["sampler_name"] != "er_sde" ||
		sampler["scheduler"] != sampling.ScheduleBongTangent || len(recording.Texts["607"]) != 1 || len(recording.Texts["608"]) != 1 {
		t.Fatalf("the recording set %v over a sampler of %v with prompts %v; want the section at 1024 x 576 from the recorder's picture", recording.Set, sampler, recording.Texts)
	}
	const width, height = 1024, 576
	picture := testcheck.Must(png.Decode(bytes.NewReader(ultraPictureBytes(t))))(t)
	levels := testcheck.Must(media.ScaleLikeComfyUI(picture, width, height))(t)
	unit := make([]float32, len(levels))
	for index, level := range levels {
		unit[index] = float32(level) / media.UnitLevels
	}
	estimate := testcheck.Must(vitencoder.LoadDepthAnything(filepath.Join(testutil.ModelArtifactDir(t, "depth-anything-v2-small"), "depth_anything_v2_vits.pth")))(t)
	depth, mapHeight, mapWidth, err := estimate.DepthMap(unit, height, width, int(estimator["resolution"].(float64)))
	testcheck.NoError(t, err)
	gray := image.NewGray(image.Rect(0, 0, mapWidth, mapHeight))
	copy(gray.Pix, depth)
	var depthPNG bytes.Buffer
	testcheck.NoError(t, png.Encode(&depthPNG, gray))
	guidance := float32(sampler["cfg"].(float64))
	strength := float32(control["strength"].(float64))
	end := control["end_percent"].(float64)
	none := []int{}
	return Request{Prompt: recording.Texts["607"][0], NegativePrompt: recording.Texts["608"][0], GuidanceScale: &guidance, Width: width, Height: height,
		Steps: int(sampler["steps"].(float64)), Seed: int64(sampler["seed"].(float64)), Adapter: "moss-control", SkipBlocks: &none,
		Schedule: &sampling.Schedule{Sampler: sampling.SamplerERSDE, Name: sampling.ScheduleBongTangent},
		Control:  &DepthControl{Map: base64.StdEncoding.EncodeToString(depthPNG.Bytes()), Strength: &strength, Start: control["start_percent"].(float64), End: &end}}, depthPNG.Bytes()
}
