//go:build modeltest && windows

package latentimage

import (
	"context"
	"encoding/base64"
	"os"
	"slices"
	"syscall"
	"testing"
	"time"
	"unsafe"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/landing"
	"overgo/internal/processmeasure"
	"overgo/internal/quietmachine"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// workflowSection is one of the MOSS Ultra workflow's sections as the workflow orders them: its recordings' folder name
// and how its request is asked.
type workflowSection struct {
	slug    string
	request func(t *testing.T, recording comfySection) Request
}

// workflowTextToImage is the text to image section as the workflow saves it, on its empty prompt, recorded at 1024 x
// 576 (pass-1, pass-2): its prompts as its preview nodes show them, its sampler and its sharpening (textSection). The
// golden floor holds the section on a real prompt instead (moss-real-prompt).
func workflowTextToImage(t *testing.T, recording comfySection) Request {
	t.Helper()
	if recording.Set["670.megapixels"] != 0.5625 {
		t.Fatalf("the recording set %v; want the canvas at 1024 x 576", recording.Set)
	}
	return textSection(t, recording, "598", "436", "592", "599")
}

// workflowImageToImage is the image to image section as the workflow saves it, at its denoise of 1 from the recorder's
// picture at 1024 x 576 (pass-1, pass-2): it starts from the noise alone, at sigma 1, the same run as text to image.
func workflowImageToImage(t *testing.T, recording comfySection) Request {
	t.Helper()
	if recording.Set["428.image"] != "overgo-recorder-input.png" || recording.Set["680.megapixels"] != 0.5625 {
		t.Fatalf("the recording set %v; want the recorder's picture at 1024 x 576", recording.Set)
	}
	request := textSection(t, recording, "506", "585", "586", "603")
	request.Image, request.Denoise = base64.StdEncoding.EncodeToString(ultraPictureBytes(t)), recording.inputs("506")["denoise"].(float64)
	return request
}

// workflowSections are the MOSS Ultra workflow's six sections in its own order, its groups' order in workflow.json, each
// asked as ComfyUI's run asked it (moss-workflow-clock): text to image and image to image as the workflow saves them
// (workflowTextToImage, workflowImageToImage), the Turbo sections drawn from a picture as the Turbo floors ask them
// (turboCases), text to image Turbo through its LoRA loader 341, and depth control from its picture, its map estimated
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
		{"text-to-image", workflowTextToImage},
		{"image-to-image", workflowImageToImage},
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

// processUse is what this process has spent by a moment: its processor time, user and kernel, and the bytes it has
// read through its files (GetProcessTimes, GetProcessIoCounters).
type processUse struct {
	cpu  time.Duration
	read uint64
}

// processIOCounters is GetProcessIoCounters', in IO_COUNTERS' order.
var processIOCounters = syscall.NewLazyDLL("kernel32.dll").NewProc("GetProcessIoCounters")

// readProcessUse reads what this process has spent by now.
func readProcessUse(t *testing.T) processUse {
	t.Helper()
	self := testcheck.Must(syscall.GetCurrentProcess())(t)
	var creation, exit, kernel, user syscall.Filetime
	testcheck.NoError(t, syscall.GetProcessTimes(self, &creation, &exit, &kernel, &user))
	var counters struct{ readOperations, writeOperations, otherOperations, readBytes, writeBytes, otherBytes uint64 }
	if ok, _, err := processIOCounters.Call(uintptr(self), uintptr(unsafe.Pointer(&counters))); ok == 0 {
		t.Fatalf("GetProcessIoCounters: %v", err)
	}
	// A FILETIME span counts 100-nanosecond intervals.
	span := func(at syscall.Filetime) time.Duration {
		return time.Duration(uint64(at.HighDateTime)<<32|uint64(at.LowDateTime)) * 100
	}
	return processUse{cpu: span(kernel) + span(user), read: counters.readBytes}
}

// TestTheUltraWorkflowIsTimedInItsOrder reads the MOSS Ultra workflow's whole time on overgo against ComfyUI's
// (moss-workflow-time, moss-workflow-clock): its six sections, each asked as ComfyUI's run asked it, run in the
// workflow's order from one session as the served image capability keeps one (a session per picture extent, reset for
// each request), twice, each section's wall from its request to its picture beside its recording's milliseconds, the
// first run beside ComfyUI's first (pass-1) and the second beside its second (pass-2), and each section's sampling a
// pass at a time. Each section's parts are read beside its wall (moss-first-run/quiet): the session's load or reset,
// the prompt's preparation, the sampling and the decode, each with the processor time and the bytes read from files it
// took, so the first run's time is split where it goes. The campaign's number is the second run's whole against
// ComfyUI's pass-2, appended to the record workflowRecordVariable names with the commit and changed paths read as the
// timing starts, and refused where the checkout's commit moved while it ran (appendWorkflowTiming). A timing that
// records holds the machine quiet while it times (timedQuietly): the quiet machine alone, then the card's turn and the
// test turn, which every lane's step takes, so no commit, workload or lane runs beside it and no wait is timed; one that
// records nothing runs in its runner's turns as any test does. Each section draws a picture.
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
	var timing workflowTiming
	runs := func() error {
		timing.At = time.Now().UTC().Format(time.RFC3339)
		for run, recorded := range []string{"pass-1", "pass-2"} {
			times := workflowRunTime{Recorded: recorded}
			var overgo, comfy time.Duration
			for _, section := range sections {
				recording := readComfySection(t, recorded, section.slug)
				request := section.request(t, recording)
				if section.slug == "depth-control" && (request.Control == nil || request.Control.Picture == "" || request.Control.Map != "") {
					t.Fatal("depth control is asked with a map made before it; want it from its picture, as ComfyUI's run asked it")
				}
				var parts []workflowPart
				start, use := time.Now(), readProcessUse(t)
				began, used := start, use
				part := func(name string) {
					now, spent := time.Now(), readProcessUse(t)
					parts = append(parts, workflowPart{Part: name, Milliseconds: now.Sub(began).Milliseconds(), CPUMs: (spent.cpu - used.cpu).Milliseconds(), ReadBytes: spent.read - used.read})
					began, used = now, spent
				}
				if generator == nil || generator.Reset(ctx, request) != nil {
					if generator != nil {
						closeReporting(t, context.WithoutCancel(ctx), generator)
					}
					generator = testcheck.Must(LoadGenerator(ctx, root, profile, request))(t)
				}
				part("load")
				session := testcheck.Must(generator.prepare(ctx, request))(t)
				part("prepare")
				before := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes
				sampling := time.Now()
				testcheck.Must(generator.integrate(ctx, session))(t)
				sampled := time.Since(sampling)
				copied := testcheck.Must(generator.rotary.runtime.Stats(ctx))(t).Execution.HostToDeviceBytes - before
				part("sampling")
				// The decode part is read as its own parts besides (png-level): "decode: decoder compiled", "decode: decoded
				// on the card" and "decode: finished", each a wall within the decode part's, not beside it.
				staged, stages := processmeasure.WithStages(ctx)
				picture := testcheck.Must(generator.decode(staged, session))(t)
				part("decode")
				for _, within := range testcheck.Must(stages.Parts())(t) {
					parts = append(parts, workflowPart{Part: "decode: " + within.Name, Milliseconds: within.Wall.Milliseconds()})
				}
				wall, spent := time.Since(start), readProcessUse(t)
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
					SamplingMs: sampled.Milliseconds(), Passes: passes, PlacedBytes: copied, ComfyMs: recordedWall.Milliseconds(),
					CPUMs: (spent.cpu - use.cpu).Milliseconds(), ReadBytes: spent.read - use.read, Parts: parts})
				t.Logf("run %d, %s: %v from request to picture (sampling %v with %.2f GiB placed on the card, %d passes at %v a pass), ComfyUI's %s %v",
					run+1, section.slug, wall.Round(time.Millisecond), sampled.Round(time.Millisecond), float64(copied)/(1<<30), passes,
					(sampled / time.Duration(max(passes, 1))).Round(time.Millisecond), recorded, recordedWall)
				for _, each := range parts {
					t.Logf("run %d, %s, %s: %v, %v of processor time, %.2f GiB read", run+1, section.slug, each.Part,
						time.Duration(each.Milliseconds)*time.Millisecond, time.Duration(each.CPUMs)*time.Millisecond, float64(each.ReadBytes)/(1<<30))
				}
			}
			times.Milliseconds, times.ComfyMs = overgo.Milliseconds(), comfy.Milliseconds()
			timing.Runs = append(timing.Runs, times)
			t.Logf("run %d: the workflow's six sections took %v on overgo, ComfyUI's %s %v", run+1, overgo.Round(time.Millisecond), recorded, comfy)
		}
		return nil
	}
	if path != "" {
		const purpose = "the workflow's timing (moss-workflow-clock)"
		turns := func(work func() error) error {
			return landing.HoldTestTurnEach(purpose, func(inner func() error) error { return landing.HoldLaneTurn(purpose, inner) }, []func() error{work})
		}
		waited := testcheck.Must(timedQuietly(ctx, quietmachine.Name, os.Stderr, turns, runs))(t)
		t.Logf("the timing waited %v for the quiet machine, which it then held alone with the card's turn and the test turn", waited.Round(time.Millisecond))
		testcheck.NoError(t, appendWorkflowTiming(repository, path, source, timing))
		t.Logf("appended the timing of %s to %s", source.commit, path)
	} else {
		testcheck.NoError(t, runs())
	}
}
