//go:build modeltest && windows

package latentimage

import (
	"context"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// stepCopiesSteps are the two step counts TestMOSSStepsCopyNothingToTheHost draws, whose difference is two steps of
// ER-SDE's second and third orders: what a request copies once (its noise, its start latent, its control) cancels.
var stepCopiesSteps = [2]int{2, 4}

// stepCopiesStepBytes bound what a step may copy to the card: each pass's timestep embedding and the step's delta, a
// few kilobytes, where a latent is over a megabyte.
const stepCopiesStepBytes = 64 << 10

// TestMOSSStepsCopyNothingToTheHost holds a MOSS sampling to keeping its latent on the card (moss-step-copies): text to
// image and depth control, each asked as the workflow's pass-2 recording asks it at the workflow's size, drawn once
// to place their weights and then for 2 steps and for 4, copy nothing more to the host for the two steps more, their
// ER-SDE steps, predictions, divided differences and noise all on the card (sampling.ERSDEDevice), and at most
// stepCopiesStepBytes a step more to the card, depth control's control and the zeros outside its window held there
// once. Counted by the driver's copy statistics around each sampling. Before, each ER-SDE step copied its latent and
// prediction to the host and the stepped latent and its noise between them, and each depth control pass uploaded the
// control, 2.4 MB: 5.4 to 8.8 ms of copies a pass against ComfyUI's 0.07.
func TestMOSSStepsCopyNothingToTheHost(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	ultraDepth(t, root)
	profile := testcheck.Must(ResolveProfile(root))(t)
	var generator *Generator
	defer func() {
		if generator != nil {
			closeReporting(t, context.WithoutCancel(ctx), generator)
		}
	}()
	for _, section := range workflowSections(t) {
		if section.slug != "text-to-image" && section.slug != "depth-control" {
			continue
		}
		recording := readComfySection(t, "pass-2", section.slug)
		var down, up [2]uint64
		// A first draw places the section's weights and adapters; the two after it are read.
		for run, steps := range append([]int{stepCopiesSteps[0]}, stepCopiesSteps[:]...) {
			request := section.request(t, recording)
			if section.slug == "depth-control" {
				request, _ = depthControlSection(t, recording)
			}
			request.Steps = steps
			if generator == nil || generator.Reset(ctx, request) != nil {
				if generator != nil {
					closeReporting(t, context.WithoutCancel(ctx), generator)
				}
				generator = testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
			}
			session := testcheck.Must(generator.prepare(ctx, request))(t)
			statistics := func() driver.ExecutionStats {
				var read driver.ExecutionStats
				testcheck.NoError(t, generator.rotary.runtime.Do(ctx, func(state *device.State) error {
					read = state.Driver.ExecutionStats()
					return nil
				}))
				return read
			}
			before := statistics()
			testcheck.Must(generator.integrate(ctx, session))(t)
			after := statistics()
			if run > 0 {
				down[run-1], up[run-1] = after.DeviceToHostBytes-before.DeviceToHostBytes, after.HostToDeviceBytes-before.HostToDeviceBytes
			}
			testcheck.Must(generator.decode(ctx, session))(t)
		}
		more := uint64(stepCopiesSteps[1] - stepCopiesSteps[0])
		t.Logf("%s: %d steps copied %d bytes to the host and %d to the card, %d steps %d and %d", section.slug,
			stepCopiesSteps[0], down[0], up[0], stepCopiesSteps[1], down[1], up[1])
		if down[1] != down[0] {
			t.Errorf("%s's %d steps more copied %d bytes more to the host; a sampling keeps its latent on the card", section.slug, more, int64(down[1]-down[0]))
		}
		if up[1] > up[0]+more*stepCopiesStepBytes {
			t.Errorf("%s's %d steps more copied %d bytes more to the card, past %d a step", section.slug, more, int64(up[1]-up[0]), stepCopiesStepBytes)
		}
	}
}
