//go:build modeltest && windows

package latentimage

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
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

// workflowSectionTime is one section's figures in a run of the workflow: its wall from request to picture, its
// sampling, its passes, the bytes it placed on the card, and its recording's wall.
type workflowSectionTime struct {
	Section      string `json:"section"`
	Milliseconds int64  `json:"ms"`
	SamplingMs   int64  `json:"sampling_ms"`
	Passes       int    `json:"passes"`
	PlacedBytes  uint64 `json:"placed_bytes"`
	ComfyMs      int64  `json:"comfy_ms"`
}

// workflowRunTime is one run of the six sections, the first beside ComfyUI's pass-1 and the second beside its pass-2.
type workflowRunTime struct {
	Recorded     string                `json:"recorded"`
	Milliseconds int64                 `json:"ms"`
	ComfyMs      int64                 `json:"comfy_ms"`
	Sections     []workflowSectionTime `json:"sections"`
}

// workflowTiming is one timing of the workflow as its record keeps it: the commit it timed, the paths the tree held
// changed from it, when, and its two runs.
type workflowTiming struct {
	Commit  string            `json:"commit"`
	Changed []string          `json:"changed"`
	At      string            `json:"at"`
	Runs    []workflowRunTime `json:"runs"`
}

// TestTheUltraWorkflowIsTimedInItsOrder reads the MOSS Ultra workflow's whole time on overgo against ComfyUI's
// (moss-workflow-time, moss-workflow-clock): its six sections, each asked as ComfyUI's run asked it, run in the
// workflow's order from one session as the served image capability keeps one (a session per picture extent, reset for
// each request), twice, each section's wall from its request to its picture beside its recording's milliseconds, the
// first run beside ComfyUI's first (pass-1) and the second beside its second (pass-2), and each section's sampling a
// pass at a time; the test holds the card's turn alone. The campaign's number is the second run's whole against
// ComfyUI's pass-2, appended with the commit to the record workflowRecordVariable names. Each section draws a picture.
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
	if path := os.Getenv(workflowRecordVariable); path != "" {
		appendWorkflowTiming(t, path, timing)
	}
}

// appendWorkflowTiming appends timing to the record at path in the repository, one JSON line, with the commit the
// repository holds and the paths its tree holds changed from it.
func appendWorkflowTiming(t *testing.T, path string, timing workflowTiming) {
	t.Helper()
	repo := testutil.RepoRoot(t)
	git := func(arguments ...string) string {
		t.Helper()
		command := exec.Command("git", arguments...)
		command.Dir = repo
		return string(testcheck.Must(command.Output())(t))
	}
	timing.Commit = strings.TrimSpace(git("rev-parse", "HEAD"))
	timing.Changed = []string{}
	for line := range strings.Lines(git("status", "--porcelain", "--untracked-files=no")) {
		if changed := strings.TrimSpace(line[min(3, len(line)):]); changed != "" && changed != filepath.ToSlash(path) {
			timing.Changed = append(timing.Changed, changed)
		}
	}
	line := append(testcheck.Must(json.Marshal(timing))(t), '\n')
	record := testcheck.Must(os.OpenFile(filepath.Join(repo, filepath.FromSlash(path)), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644))(t)
	_, err := record.Write(line)
	testcheck.NoError(t, errors.Join(err, record.Close()))
	t.Logf("appended the timing of %s to %s", timing.Commit, path)
}
