//go:build modeltest && windows

package latentimage

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// workflowSection is one of the MOSS Ultra workflow's sections as the workflow orders them: its recordings' folder name
// and how its request is asked.
type workflowSection struct {
	slug    string
	request func(t *testing.T, recording comfySection) Request
}

// workflowSections are the MOSS Ultra workflow's six sections in its own order, its groups' order in workflow.json, each
// asked as ComfyUI's run asked it (moss-workflow-clock): as the golden and Turbo floors ask them (goldenCases,
// turboCases), text to image Turbo through its LoRA loader 341, and depth control from its picture, its map estimated
// as the request is drawn, as ComfyUI's depth control run estimated it (612:438 not cached).
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
		{"depth-control", depthControlRequest},
		{"text-to-image-turbo", func(t *testing.T, recording comfySection) Request {
			return turboSection(t, recording, "331", "27", "325", "336", "341")
		}},
		{"image-to-image-turbo", named(turboCases, "image to image Turbo")},
		{"depth-control-turbo", named(turboCases, "depth control Turbo")},
	}
}

// workflowRecordVariable names the file a timing of the workflow appends its figures to, one JSON line a timing
// (docs/verification/moss-workflow-time.jsonl, moss-workflow-clock), relative to the repository; a timing not given
// one writes none.
const workflowRecordVariable = "OVERGO_WORKFLOW_RECORD"

// TestTheUltraWorkflowIsTimedInItsOrder reads the MOSS Ultra workflow's whole time on overgo against ComfyUI's
// (moss-workflow-time, moss-workflow-clock): its six sections, each asked as ComfyUI's run asked it, run in the
// workflow's order from one session as the served image capability keeps one (a session per picture extent, reset for
// each request), twice, each section's wall from its request to its picture beside its recording's milliseconds, the
// first run beside ComfyUI's first (pass-1) and the second beside its second (pass-2), and each section's sampling a
// pass at a time; the test holds the card's turn alone. The campaign's number is the second run's whole against
// ComfyUI's pass-2, appended to the record workflowRecordVariable names with the commit and changed paths read as the
// timing starts, and refused where the checkout's commit moved while it ran (appendWorkflowTiming). Each section draws
// a picture.
func TestTheUltraWorkflowIsTimedInItsOrder(t *testing.T) {
	cudatest.Require(t)
	path, repository := os.Getenv(workflowRecordVariable), testutil.RepoRoot(t)
	var source workflowSource
	if path != "" {
		source = testcheck.Must(readWorkflowSource(repository, path))(t)
	}
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
	timing := workflowTiming{At: time.Now().UTC().Format(time.RFC3339)}
	for run, recorded := range []string{"pass-1", "pass-2"} {
		times := workflowRunTime{Recorded: recorded}
		var overgo, comfy time.Duration
		for _, section := range sections {
			recording := readComfySection(t, recorded, section.slug)
			request := section.request(t, recording)
			if section.slug == "depth-control" && (request.Control == nil || request.Control.Picture == "" || request.Control.Map != "") {
				t.Fatal("depth control is asked with a map made before it; want it from its picture, as ComfyUI's run asked it")
			}
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
			times.Sections = append(times.Sections, workflowSectionTime{Section: section.slug, Milliseconds: wall.Milliseconds(),
				SamplingMs: sampled.Milliseconds(), Passes: passes, PlacedBytes: copied, ComfyMs: recordedWall.Milliseconds()})
			t.Logf("run %d, %s: %v from request to picture (sampling %v with %.2f GiB placed on the card, %d passes at %v a pass), ComfyUI's %s %v",
				run+1, section.slug, wall.Round(time.Millisecond), sampled.Round(time.Millisecond), float64(copied)/(1<<30), passes,
				(sampled / time.Duration(max(passes, 1))).Round(time.Millisecond), recorded, recordedWall)
		}
		times.Milliseconds, times.ComfyMs = overgo.Milliseconds(), comfy.Milliseconds()
		timing.Runs = append(timing.Runs, times)
		t.Logf("run %d: the workflow's six sections took %v on overgo, ComfyUI's %s %v", run+1, overgo.Round(time.Millisecond), recorded, comfy)
	}
	if path != "" {
		testcheck.NoError(t, appendWorkflowTiming(repository, path, source, timing))
		t.Logf("appended the timing of %s to %s", source.commit, path)
	}
}
