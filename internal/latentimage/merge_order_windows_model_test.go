//go:build modeltest && windows

package latentimage

import (
	"context"
	"os"
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// sampleWorkflow runs the MOSS Ultra workflow's six sections in the workflow's own order on one session, runs times, as
// they run in it (workflowSections), the three of fifty steps cut to two, and gives each section's generator after its
// picture to sampled, with the run, the section and the bytes its sampling copied to the card.
func sampleWorkflow(t *testing.T, runs int, sampled func(run int, section string, generator *Generator, copied uint64)) {
	t.Helper()
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
	for run := range runs {
		for _, section := range workflowSections(t) {
			request := section.request(t, readComfySection(t, "pass-2", section.slug))
			if !turboSlug(section.slug) {
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
			copied := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes - before
			testcheck.Must(generator.decode(ctx, session))(t)
			sampled(run, section.slug, generator, copied)
		}
	}
}

// turboSlug says a workflow section is one of its Turbo sections.
func turboSlug(slug string) bool {
	return strings.HasSuffix(slug, "-turbo")
}

// adapterFileBytes is the bytes of the files of the adapters the generator's request applies.
func adapterFileBytes(t *testing.T, generator *Generator) uint64 {
	t.Helper()
	var bytes uint64
	for _, adapter := range generator.rotary.adapters {
		bytes += uint64(testcheck.Must(os.Stat(adapter.path))(t).Size())
	}
	return bytes
}

// TestTheTurboSectionsMergeFromHeldWeightsInTheWorkflowsOrder holds the MOSS Ultra workflow's Turbo sections, in the
// workflow's own order on one session, to merging their adapters from the int8 rows the session holds on the card
// (moss-merge-order): its six sections run as the workflow runs them (sampleWorkflow), and each Turbo section, after
// the section before it, places on the card at most twice its adapters' bytes. Each had read the rows its merges patch
// from the transformer's file again, 5.32 GiB for text to image Turbo after depth control and 5.65 GiB for depth control
// Turbo, about 12 s of the workflow (moss-workflow-clock at 6395092b4): the rows were held merged by the section before,
// and only rows held as the file holds them were read on the card.
func TestTheTurboSectionsMergeFromHeldWeightsInTheWorkflowsOrder(t *testing.T) {
	cudatest.Require(t)
	sampleWorkflow(t, 1, func(_ int, section string, generator *Generator, placed uint64) {
		adapterBytes := adapterFileBytes(t, generator)
		t.Logf("%s placed %.3f GiB on the card beside its %d adapters' %.3f GiB", section, float64(placed)/(1<<30),
			len(generator.rotary.adapters), float64(adapterBytes)/(1<<30))
		if turboSlug(section) && placed > 2*adapterBytes {
			t.Errorf("%s placed %d bytes on the card after the section before it; want at most twice its adapters' %d, its merges reading the weights they patch on the card",
				section, placed, adapterBytes)
		}
	})
}

// pairBytes is the bytes in BF16 of the distinct tensors of the adapters' low-rank pairs, as the merges read them on
// the card, a fused qkv's up once for its q, k and v.
func pairBytes(t *testing.T, adapters []*mediaAdapter) uint64 {
	t.Helper()
	seen := map[string]bool{}
	var bytes uint64
	for _, adapter := range adapters {
		for _, patch := range adapter.lowRank {
			rows := patch.rows
			if patch.split {
				rows *= tensor.TripleExtent
			}
			for name, elements := range map[string]uint64{patch.up: rows * patch.rank, patch.down: patch.rank * patch.in} {
				if !seen[adapter.content+name] {
					seen[adapter.content+name] = true
					bytes += testcheck.Must(tensor.MustShape(elements).Bytes(dtype.BF16))(t)
				}
			}
		}
	}
	return bytes
}

// TestTheTurboSectionsMergeAsComfyUIPatches holds the MOSS Ultra workflow's sections, run twice in the workflow's order
// on one session as its clock runs them (moss-workflow-clock), to merging as ComfyUI's patcher merges the LoRAs its
// loaders hold (moss-merge-compute): a section whose adapters each came with an earlier section copies to the card
// under a quarter of their pairs in BF16, its merges reading the pairs the session keeps on the card under each file's
// content, so only the first section to apply an adapter reads, rounds and copies its pairs. Each request had read,
// rounded and copied every pair again, 0.95 GiB for text to image Turbo and 1.30 GiB for depth control Turbo, 3.68 and
// 5.10 s beyond their passes (moss-merge-order's clock at 19c8144b0), where ComfyUI's two sections take 1.96 and 2.21 s
// whole. Read 2026-11-20: the second run's text to image Turbo copied 0.051 GiB and depth control Turbo 0.104 GiB.
func TestTheTurboSectionsMergeAsComfyUIPatches(t *testing.T) {
	cudatest.Require(t)
	applied := map[string]bool{}
	sampleWorkflow(t, 2, func(run int, section string, generator *Generator, copied uint64) {
		adapters := generator.rotary.adapters
		pairs := pairBytes(t, adapters)
		t.Logf("run %d: %s copied %.3f GiB to the card beside its %d adapters' pairs, %.3f GiB in BF16", run+1, section,
			float64(copied)/(1<<30), len(adapters), float64(pairs)/(1<<30))
		earlier := len(adapters) > 0
		for _, adapter := range adapters {
			earlier = earlier && applied[adapter.content]
		}
		for _, adapter := range adapters {
			applied[adapter.content] = true
		}
		if earlier && copied*4 > pairs {
			t.Errorf("run %d: %s copied %d bytes to the card; want under a quarter of its adapters' pairs, %d bytes in BF16, each adapter applied by an earlier section and its pairs kept on the card",
				run+1, section, copied, pairs)
		}
	})
}
