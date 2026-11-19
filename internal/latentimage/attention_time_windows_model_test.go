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

// attentionShare is the share of a MOSS pass at 2368 x 1344 its BF16 attention took (moss-speed/do: 1.2 s of 2.8 s of
// kernels, 28.9 ms a layer).
const attentionShare = 0.44

// TestMOSSAttentionTakesHalfItsTime holds a MOSS pass at the sections' own size, 2368 x 1344, as the workflow's text
// section's preset draws it (moss-attention), to saving at least half of what its attention took in BF16: one
// generator samples the conditional pass at 2 and 6 steps in BF16 attention and as the preset asks, and a pass is
// the difference over four steps, the request's own part left beside them.
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
	asking := func(steps int, preset string) Request {
		asked := request
		asked.Steps, asked.Preset = steps, preset
		return asked
	}
	generator, loaded := testcheck.Must(LoadGenerator(ctx, root, profile, asking(2, "")))(t), true
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	sample := func(steps int, preset string) time.Duration {
		t.Helper()
		asked := asking(steps, preset)
		if !loaded {
			testcheck.NoError(t, generator.Reset(ctx, asked))
		}
		loaded = false
		session := testcheck.Must(generator.prepare(ctx, asked))(t)
		start := time.Now()
		testcheck.Must(generator.integrate(ctx, session))(t)
		wall := time.Since(start)
		testcheck.Must(generator.decode(ctx, session))(t)
		return wall
	}
	const preset = "ultra-text-to-image"
	sample(2, "")
	plain := (sample(6, "") - sample(2, "")) / 4
	served := (sample(6, preset) - sample(2, preset)) / 4
	if !generator.request.EightBitAttention {
		t.Errorf("the %s preset draws its attention in BF16; want the faster attention it asks", preset)
	}
	t.Logf("a conditional pass at %dx%d: %v in BF16 attention, %v as %s draws it", request.Width, request.Height, plain.Round(time.Millisecond), served.Round(time.Millisecond), preset)
	if saved := plain - served; float64(saved) < attentionShare/2*float64(plain) {
		t.Errorf("as %s draws it a pass saves %v of BF16's %v; want at least half its attention's share, %v",
			preset, saved.Round(time.Millisecond), plain.Round(time.Millisecond), time.Duration(attentionShare/2*float64(plain)).Round(time.Millisecond))
	}
}
