//go:build modeltest && windows

package latentimage

import (
	"context"
	"os"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestTheTurboSectionsMergeFromHeldWeightsInTheWorkflowsOrder holds the MOSS Ultra workflow's Turbo sections, in the
// workflow's own order on one session, to merging their adapters from the int8 rows the session holds on the card
// (moss-merge-order): its six sections run as the workflow runs them (workflowSections), the three of fifty steps cut
// to two, and each Turbo section, after the section before it, places on the card at most twice its adapters' bytes.
// Each had read the rows its merges patch from the transformer's file again, 5.32 GiB for text to image Turbo after
// depth control and 5.65 GiB for depth control Turbo, about 12 s of the workflow (moss-workflow-clock at 6395092b4):
// the rows were held merged by the section before, and only rows held as the file holds them were read on the card.
func TestTheTurboSectionsMergeFromHeldWeightsInTheWorkflowsOrder(t *testing.T) {
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
		request := section.request(t, readComfySection(t, "pass-2", section.slug))
		turbo := strings.HasSuffix(section.slug, "-turbo")
		if !turbo {
			request.Steps = 2
		}
		if generator == nil || generator.Reset(ctx, request) != nil {
			if generator != nil {
				closeReporting(t, context.WithoutCancel(ctx), generator)
			}
			generator = testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
		}
		session := testcheck.Must(generator.prepare(ctx, request))(t)
		before := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes
		testcheck.Must(generator.integrate(ctx, session))(t)
		placed := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes - before
		testcheck.Must(generator.decode(ctx, session))(t)
		var adapterBytes uint64
		for _, adapter := range generator.rotary.adapters {
			adapterBytes += uint64(testcheck.Must(os.Stat(adapter.path))(t).Size())
		}
		t.Logf("%s placed %.3f GiB on the card beside its %d adapters' %.3f GiB", section.slug, float64(placed)/(1<<30),
			len(generator.rotary.adapters), float64(adapterBytes)/(1<<30))
		if turbo && placed > 2*adapterBytes {
			t.Errorf("%s placed %d bytes on the card after the section before it; want at most twice its adapters' %d, its merges reading the weights they patch on the card",
				section.slug, placed, adapterBytes)
		}
	}
}
