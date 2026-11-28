//go:build modeltest && windows

package latentimage

import (
	"context"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/media"
	"overgo/internal/processmeasure"
	"overgo/internal/testcheck"
)

// stepSample is one sampling of a request: its wall, the bytes it copied to the card, its kernel launches and when, from
// the sampling's start, each of its steps completed.
type stepSample struct {
	wall       time.Duration
	hostToCard uint64
	launches   uint64
	completed  []time.Duration
}

// step is the median time between the sampling's completed steps after its first: a step read within one draw, which
// another draw's pace does not move.
func (s stepSample) step() time.Duration {
	between := make([]time.Duration, 0, len(s.completed))
	for index := 1; index < len(s.completed); index++ {
		between = append(between, s.completed[index]-s.completed[index-1])
	}
	if len(between) == 0 {
		return 0
	}
	slices.Sort(between)
	return between[len(between)/2]
}

// TestWhereAMossStepsTimeGoes reads where a MOSS sampling's time goes at the sections' own size, 2368 x 1344
// (moss-speed/do), on the MOSS Ultra workflow's files and its own-size text section, from one generator sampling the
// guided request at 2 and 6 steps and at guidance 1 its conditional pass alone at 6 steps: a guided step and a pass of
// the denoiser, each the time between completed steps of one draw (processmeasure.WithSteps), as ComfyUI's step beside
// them is the time between its progress events; the first step, which binds the request; and the kernel launches and
// bytes to the card a step adds. A guided step runs two passes, so it takes longer than one. What a guided step spends
// beside its two passes would be one draw's step less another's passes, so it is not read. The step had been the difference of the 6- and 2-step walls over four and a request's own part
// what that left of the 2-step wall, which read -717 ms at efa59bd59, its draws beside another process on the card
// (SQA-520); before that a sampling had been read as one step in six, which charged each step a sixth of the weights'
// 9.5 GB upload (10.9 s a step).
func TestWhereAMossStepsTimeGoes(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	profile := testcheck.Must(ResolveProfile(root))(t)
	recording := readComfySection(t, "own-size", "text-to-image")
	request := textSection(t, recording, "598", "436", "592", "599")
	request.Width, request.Height, request.Canvas = 0, 0, &media.Canvas{Aspect: "16:9", Megapixels: 3, Multiple: 32}
	request = testcheck.Must(request.Sized(CatalogIn(root)))(t)
	// ComfyUI's step is read before the generator loads, so a recording that cannot give one costs no draw (SQA-477).
	comfy := recording.stepTime(t, "598")
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
		var completed []time.Duration
		start := time.Now()
		stepping := processmeasure.WithSteps(ctx, func(int, int) { completed = append(completed, time.Since(start)) })
		testcheck.Must(generator.integrate(stepping, session))(t)
		wall := time.Since(start)
		after := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution
		testcheck.Must(generator.decode(ctx, session))(t)
		if len(completed) != steps {
			t.Fatalf("a %d-step sampling told %d completed steps", steps, len(completed))
		}
		return stepSample{wall: wall, hostToCard: after.HostToDeviceBytes - before.HostToDeviceBytes, launches: after.KernelLaunches - before.KernelLaunches, completed: completed}
	}
	one := float32(1)
	sample(2, nil)
	two, six, alone := sample(2, nil), sample(6, nil), sample(6, &one)
	step, pass := six.step(), alone.step()
	t.Logf("%dx%d: a first step %v of a %v sampling, steps completing at %v, copying %.2f GiB to the card (%.2f GiB in 6 steps); a guided step %v, %d kernel launches; a pass %v; ComfyUI's guided step %v",
		request.Width, request.Height, six.completed[0].Round(time.Millisecond), six.wall.Round(time.Millisecond), six.completed,
		float64(two.hostToCard)/(1<<30), float64(six.hostToCard)/(1<<30), step.Round(time.Millisecond), (six.launches-two.launches)/4,
		pass.Round(time.Millisecond), comfy.Round(time.Millisecond))
	if !(pass > 0 && step > pass) {
		t.Fatalf("a guided step reads %v and a pass %v; want a pass above nothing and a guided step, two passes, above it", step, pass)
	}
}
