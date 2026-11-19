//go:build modeltest && windows

package latentimage

import (
	"context"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// workflowSection is one of the MOSS Ultra workflow's sections as the workflow orders them: its recordings' folder name
// and how its request is asked.
type workflowSection struct {
	slug    string
	request func(t *testing.T, recording comfySection) Request
}

// workflowSections are the MOSS Ultra workflow's six sections in its own order, its groups' order in workflow.json, each
// asked as the golden and Turbo floors ask it (goldenCases, turboCases), text to image Turbo through its LoRA loader 341.
func workflowSections(t *testing.T) []workflowSection {
	t.Helper()
	named := func(cases []goldenCase, name string) func(*testing.T, comfySection) Request {
		at := slices.IndexFunc(cases, func(golden goldenCase) bool { return golden.name == name })
		if at < 0 {
			t.Fatalf("no case is named %q", name)
		}
		return cases[at].request
	}
	return []workflowSection{
		{"text-to-image", named(goldenCases, "text to image")},
		{"image-to-image", named(goldenCases, "image to image")},
		{"depth-control", named(goldenCases, "depth control")},
		{"text-to-image-turbo", func(t *testing.T, recording comfySection) Request {
			return turboSection(t, recording, "331", "27", "325", "336", "341")
		}},
		{"image-to-image-turbo", named(turboCases, "image to image Turbo")},
		{"depth-control-turbo", named(turboCases, "depth control Turbo")},
	}
}

// TestTheUltraWorkflowIsTimedInItsOrder reads the MOSS Ultra workflow's whole time on overgo against ComfyUI's
// (moss-workflow-time): its six sections, each asked as its recording asks it, run in the workflow's order from one
// session as the served image capability keeps one (a session per picture extent, reset for each request), twice, each
// section's wall from its request to its picture beside its recording's milliseconds, the first run beside ComfyUI's
// first (pass-1) and the second beside its second (pass-2), and each section's sampling a pass at a time. The campaign's
// number is the whole; the speed rows after it are ordered by what each saves on it (the owner's review of
// 2026-11-19). Each section draws a picture.
func TestTheUltraWorkflowIsTimedInItsOrder(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	root := ultraModelRoot(t)
	ultraDepth(t, root)
	profile := testcheck.Must(ResolveProfile(root))(t)
	sections := workflowSections(t)
	var generator *Generator
	defer func() {
		if generator != nil {
			closeReporting(t, context.WithoutCancel(ctx), generator)
		}
	}()
	for run, recorded := range []string{"pass-1", "pass-2"} {
		var overgo, comfy time.Duration
		for _, section := range sections {
			recording := readComfySection(t, recorded, section.slug)
			request := section.request(t, recording)
			start := time.Now()
			if generator == nil || generator.Reset(ctx, request) != nil {
				if generator != nil {
					closeReporting(t, context.WithoutCancel(ctx), generator)
				}
				generator = testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
			}
			session := testcheck.Must(generator.prepare(ctx, request))(t)
			before := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes
			sampling := time.Now()
			testcheck.Must(generator.integrate(ctx, session))(t)
			sampled := time.Since(sampling)
			copied := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes - before
			picture := testcheck.Must(generator.decode(ctx, session))(t)
			wall := time.Since(start)
			if len(picture.Data) == 0 {
				t.Fatalf("run %d: %s drew no picture", run+1, section.slug)
			}
			passes := generator.request.Steps
			if generator.rotary.guidance != 1 {
				passes *= 2
			}
			recordedWall := time.Duration(recording.Milliseconds) * time.Millisecond
			overgo, comfy = overgo+wall, comfy+recordedWall
			t.Logf("run %d, %s: %v from request to picture (sampling %v with %.2f GiB placed on the card, %d passes at %v a pass), ComfyUI's %s %v",
				run+1, section.slug, wall.Round(time.Millisecond), sampled.Round(time.Millisecond), float64(copied)/(1<<30), passes,
				(sampled / time.Duration(max(passes, 1))).Round(time.Millisecond), recorded, recordedWall)
		}
		t.Logf("run %d: the workflow's six sections took %v on overgo, ComfyUI's %s %v", run+1, overgo.Round(time.Millisecond), recorded, comfy)
	}
}
