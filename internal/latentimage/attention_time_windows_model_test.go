//go:build modeltest && windows

package latentimage

import (
	"context"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// attentionShare is the share of a MOSS pass at 2368 x 1344 its native BF16 attention took (moss-speed/do: 1.2 s of
// 2.8 s of kernels, 28.9 ms a layer).
const attentionShare = 0.44

// samePicture is the least agreement, in dB, a sampling with its attention in registers keeps with the native kernel's
// at the same request and seed, latent and picture, read as the golden floors read theirs: the same picture, where the
// workflow's own recording keeps 1.55 dB of latent and 15.11 dB of picture.
const samePicture = 30.0

// TestMOSSAttentionTakesHalfItsTime holds MOSS's sampling at the sections' own size, 2368 x 1344, with its attention
// in registers (samplingScoresInRegisters, moss-attention), to saving at least half what the native BF16 kernel's
// attention took of a pass, and to drawing the same picture as the native kernel at the same request and seed: one
// generator samples the conditional pass at 2 and 6 steps with each kernel, a pass being the difference over four
// steps, and the 6-step samplings' final latents and pictures agree to samePicture. Golden floors 3 dB under readings
// near 1.5 dB would pass a kernel that drew another picture (the owner's review of 2026-11-18).
func TestMOSSAttentionTakesHalfItsTime(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	recording := readComfySection(t, "own-size", "text-to-image")
	request := textSection(t, recording, "598", "436", "592", "599")
	request.Width, request.Height, request.Canvas = 0, 0, &media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}
	request = testcheck.Must(request.Sized(root))(t)
	one := float32(1)
	request.GuidanceScale = &one
	served := samplingScoresInRegisters
	t.Cleanup(func() { samplingScoresInRegisters = served })
	asking := func(steps int) Request {
		asked := request
		asked.Steps = steps
		return asked
	}
	generator, loaded := testcheck.Must(LoadGenerator(ctx, root, profile, asking(2)))(t), true
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	type drawn struct {
		wall    time.Duration
		latent  []float32
		picture []byte
	}
	sample := func(steps int, registers bool) drawn {
		t.Helper()
		samplingScoresInRegisters = registers
		asked := asking(steps)
		if !loaded {
			testcheck.NoError(t, generator.Reset(ctx, asked))
		}
		loaded = false
		session := testcheck.Must(generator.prepare(ctx, asked))(t)
		start := time.Now()
		testcheck.Must(generator.integrate(ctx, session))(t)
		wall := time.Since(start)
		latent := testcheck.Must(generator.rotary.latent.outputs.CopyToHost(ctx, generator.rotary.latent.node))(t)
		picture := testcheck.Must(generator.decode(ctx, session))(t)
		return drawn{wall: wall, latent: latent.Data, picture: picture.Data}
	}
	sample(2, false)
	nativeSix := sample(6, false)
	native := (nativeSix.wall - sample(2, false).wall) / 4
	registersSix := sample(6, true)
	registers := (registersSix.wall - sample(2, true).wall) / 4
	latentAgreement, pictureAgreement := decibelsOverSquares(registersSix.latent, nativeSix.latent), pictureDecibels(t, registersSix.picture, nativeSix.picture)
	t.Logf("a conditional pass at %dx%d: %v with the native attention, %v with it in registers; the same seed's latent agrees to %.2f dB, its picture to %.2f dB",
		request.Width, request.Height, native.Round(time.Millisecond), registers.Round(time.Millisecond), latentAgreement, pictureAgreement)
	if saved := native - registers; float64(saved) < attentionShare/2*float64(native) {
		t.Errorf("with its attention in registers a pass saves %v of the native %v; want at least half its attention's share, %v",
			saved.Round(time.Millisecond), native.Round(time.Millisecond), time.Duration(attentionShare/2*float64(native)).Round(time.Millisecond))
	}
	if !(latentAgreement >= samePicture) || !(pictureAgreement >= samePicture) {
		t.Errorf("with its attention in registers the same seed's latent agrees to %.2f dB and its picture to %.2f dB with the native kernel's; want the same picture, at least %.0f dB",
			latentAgreement, pictureAgreement, samePicture)
	}
}
