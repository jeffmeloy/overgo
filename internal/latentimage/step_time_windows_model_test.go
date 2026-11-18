//go:build modeltest && windows

package latentimage

import (
	"context"
	"math"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/testcheck"
)

// stepSample is one sampling of a request: its wall and the bytes it copied to the card.
type stepSample struct {
	wall       time.Duration
	hostToCard uint64
	launches   uint64
}

// TestWhereAMossStepsTimeGoes reads where a MOSS sampling's time goes at the sections' own size, 2368 x 1344
// (moss-speed/do), on the MOSS Ultra workflow's files and its own-size text section, from one generator sampling the
// guided request at 2, 4 and 6 steps and at guidance 1 its conditional pass alone at 6 steps: a request's own part (the
// denoiser's weights bound to the card, which the 2-step and 6-step readings leave beside the steps), a guided step,
// a pass of the denoiser and what a guided step spends beside its two passes, the bytes a request copies to the card,
// and ComfyUI's step from the recording beside them. The parts account for the whole: the request's part and four
// guided steps read the 4-step sampling to within a tenth. A sampling had been read as one step in six, which charged
// each step a sixth of the weights' 9.5 GB upload (10.9 s a step).
func TestWhereAMossStepsTimeGoes(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	recording := readComfySection(t, "own-size", "text-to-image")
	request := textSection(t, recording, "598", "436", "592", "599")
	request.Width, request.Height, request.Canvas = 0, 0, &media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}
	request = testcheck.Must(request.Sized(root))(t)
	asking := func(steps int, guidance *float32) Request {
		asked := request
		asked.Steps = steps
		if guidance != nil {
			asked.GuidanceScale = guidance
		}
		return asked
	}
	// The generator is loaded for the first sampling and reset for each after it.
	generator, loaded := testcheck.Must(LoadGenerator(ctx, root, profile, asking(2, nil)))(t), true
	defer closeReporting(t, context.WithoutCancel(ctx), generator)
	sample := func(steps int, guidance *float32) stepSample {
		t.Helper()
		asked := asking(steps, guidance)
		if !loaded {
			testcheck.NoError(t, generator.Reset(ctx, asked))
		}
		loaded = false
		session := testcheck.Must(generator.prepare(ctx, asked))(t)
		before := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution
		start := time.Now()
		testcheck.Must(generator.integrate(ctx, session))(t)
		wall := time.Since(start)
		after := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution
		testcheck.Must(generator.decode(ctx, session))(t)
		return stepSample{wall: wall, hostToCard: after.HostToDeviceBytes - before.HostToDeviceBytes, launches: after.KernelLaunches - before.KernelLaunches}
	}
	one := float32(1)
	sample(2, nil)
	two, four, six, alone := sample(2, nil), sample(4, nil), sample(6, nil), sample(6, &one)
	step := (six.wall - two.wall) / 4
	own := two.wall - 2*step
	pass := (six.wall - alone.wall) / 6
	stamps := recording.Progress["598"]
	if len(stamps) < 2 {
		t.Fatalf("the own-size recording stamps %d of its sampler's steps", len(stamps))
	}
	comfy := time.Duration(stamps[len(stamps)-1].Microseconds-stamps[0].Microseconds) * time.Microsecond / time.Duration(len(stamps)-1)
	t.Logf("%dx%d: a request's own part %v, copying %.2f GiB to the card (%.2f GiB in 6 steps); a guided step %v, %d kernel launches; a pass %v, and beside a guided step's two passes %v; ComfyUI's guided step %v",
		request.Width, request.Height, own.Round(time.Millisecond), float64(two.hostToCard)/(1<<30), float64(six.hostToCard)/(1<<30), step.Round(time.Millisecond),
		(six.launches-two.launches)/4, pass.Round(time.Millisecond), (step - 2*pass).Round(time.Millisecond), comfy.Round(time.Millisecond))
	if !(own > 0 && step > 0 && pass > 0) {
		t.Fatalf("the readings part as a request's own %v, a step %v and a pass %v; want each above nothing", own, step, pass)
	}
	if predicted := own + 4*step; math.Abs(float64(four.wall-predicted)) > float64(four.wall)/10 {
		t.Errorf("a request's part and four steps read %v; the 4-step sampling took %v, so the parts do not account for the whole", predicted, four.wall)
	}
}
