//go:build windows

package latentimage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"image/png"
	"os"
	"testing"
	"time"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestImagePublicationStreamsEncodedArtifact streams Krea's picture for one request through publication and holds its
// geometry and its PNG, and reports its stages' walls. Its steps and its picture are held to
// the model's own pipeline, not here: the mend of 56a5820a7 numbered the prompt's closing tokens and set a distilled
// checkpoint's shift as the model's own code does, so the earlier implementation's step statistics and decoded picture,
// which this test had held, describe another picture (cycle-lanes/krea-baseline). internal/adaptiveparity
// TestServedPictureMatchesTheModelsOwnPipeline holds the served picture to the model's own code, and
// internal/mediacapability TestFrozenKreaGeneration the frozen request's to its accepted sample. Its wall is reported and
// refuses nothing: one request's wall read in a run of other models' tests stood at 19.6, 26.4 and 44 s as the file
// cache held the text encoder or not, against the earlier implementation's 25.4 s, and a timing taken beside other work
// is no measurement; the scorecard's timed cells measure Krea's first output.
func TestImagePublicationStreamsEncodedArtifact(t *testing.T) {
	requireLongTest(t)
	if os.Getenv("OVERGO_KREA_BASELINE") != "1" {
		testskip.NotApplicable(t, "set OVERGO_KREA_BASELINE=1 to measure the real Krea pipeline")
	}
	request := Request{
		Prompt: "a red fox licking a vanilla ice cream cone in snow",
		Width:  256, Height: 256, Steps: 8, Seed: 42, DynamicShiftMu: 1.15,
	}
	ctx := t.Context()
	totalStart := time.Now()
	profile := kreaProfileOrSkip(t)
	generator := testcheck.Must(LoadGenerator(ctx, kreaModelDir(t), profile, request))(t)
	defer closeReporting(t, ctx, generator)
	loadWall := time.Since(totalStart)

	prepareStart := time.Now()
	session := testcheck.Must(generator.prepare(ctx, request))(t)
	prepareWall := time.Since(prepareStart)
	var integrateWall time.Duration
	for step := range generator.schedule.Steps {
		started := time.Now()
		testcheck.NoError(t, generator.pipeline.Advance(ctx, generator.schedule.Sigmas[step], generator.schedule.Deltas[step]))
		integrateWall += time.Since(started)
	}
	generator.integrated = true
	decodeStart := time.Now()
	image := testcheck.Must(generator.decode(ctx, session))(t)
	decodeWall := time.Since(decodeStart)
	if image.Channels != 3 || image.Height != 256 || image.Width != 256 || len(image.Data) == 0 || image.MediaType != "image/png" {
		t.Fatalf("image geometry = %d/%dx%d bytes=%d type=%q", image.Channels, image.Width, image.Height, len(image.Data), image.MediaType)
	}
	if len(image.Data) >= image.Channels*image.Width*image.Height*4 {
		t.Fatalf("PNG bytes=%d, raw float bytes=%d", len(image.Data), image.Channels*image.Width*image.Height*4)
	}
	testcheck.Must(png.Decode(bytes.NewReader(image.Data)))(t)
	minimum, maximum := image.Minimum, image.Maximum
	if maximum-minimum < 0.1 {
		t.Fatalf("image is degenerate: range [%g,%g]", minimum, maximum)
	}
	var memory driver.MemoryStats
	if err := generator.pipeline.runtime.Do(ctx, func(state *device.State) error {
		memory = state.Driver.MemoryStats()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(image.Data)
	totalWall := time.Since(totalStart)
	t.Logf(
		"Krea 256 seed42 load=%.3fs prepare=%.3fs integrate=%.3fs decode+png=%.3fs total=%.3fs peak_device=%.3fGiB resident=%.3fGiB png_sha256=%s range=[%g,%g]",
		loadWall.Seconds(), prepareWall.Seconds(), integrateWall.Seconds(), decodeWall.Seconds(), totalWall.Seconds(),
		float64(memory.PeakBytes)/(1<<30), float64(generator.pipeline.ResidentBytes())/(1<<30),
		hex.EncodeToString(hash[:]), minimum, maximum,
	)
}

func TestImageSessionCacheReusesResidentRuntime(t *testing.T) {
	requireLongTest(t)
	if os.Getenv("OVERGO_KREA_BASELINE") != "1" {
		testskip.NotApplicable(t, "set OVERGO_KREA_BASELINE=1 to measure Krea residency")
	}
	request := Request{
		Prompt: "a red fox licking a vanilla ice cream cone in snow",
		Width:  256, Height: 256, Steps: 8, Seed: 42, DynamicShiftMu: 1.15,
	}
	ctx := t.Context()
	generator := testcheck.Must(LoadGenerator(ctx, kreaModelDir(t), kreaProfileOrSkip(t), request))(t)
	defer closeReporting(t, ctx, generator)
	first, firstWall := runResidentGeneration(t, ctx, generator, request)
	request.Seed++
	testcheck.NoError(t, generator.Reset(ctx, request))
	second, secondWall := runResidentGeneration(t, ctx, generator, request)
	if bytes.Equal(first.Data, second.Data) {
		t.Fatal("seed change reused request-local latent")
	}
	if secondWall >= firstWall {
		t.Fatalf("resident replay=%s cold conditioning=%s", secondWall, firstWall)
	}
	const adaptiveWall = 25_430_488_300 * time.Nanosecond
	if secondWall >= adaptiveWall {
		t.Fatalf("resident replay=%s adaptive=%s", secondWall, adaptiveWall)
	}
	t.Logf("Krea resident replay cold=%.3fs warm=%.3fs speedup=%.2fx resident=%.3fGiB",
		firstWall.Seconds(), secondWall.Seconds(), firstWall.Seconds()/secondWall.Seconds(),
		float64(generator.pipeline.ResidentBytes())/(1<<30))
}

func runResidentGeneration(
	t *testing.T,
	ctx context.Context,
	generator *Generator,
	request Request,
) (EncodedImage, time.Duration) {
	t.Helper()
	started := time.Now()
	session := testcheck.Must(generator.prepare(ctx, request))(t)
	if _, err := generator.integrate(ctx, session); err != nil {
		t.Fatal(err)
	}
	image := testcheck.Must(generator.decode(ctx, session))(t)
	return image, time.Since(started)
}
