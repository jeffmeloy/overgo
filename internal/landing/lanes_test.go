package landing

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/gosource"
	"overgo/internal/testskip"
)

// streamChild makes this test binary the command landing.run starts: it
// writes a result and a diagnostic and exits with the code it is given.
const streamChild = "OVERGO_LANDING_STREAM_CHILD"

// TestLandingRunKeepsEachStreamOwned holds the gate's command runner to one
// owner per stream: standard output comes back once, as the command wrote
// it, and a failure carries the diagnostic from standard error.
func TestLandingRunKeepsEachStreamOwned(t *testing.T) {
	if code := os.Getenv(streamChild); code != "" {
		fmt.Print("result")
		fmt.Fprint(os.Stderr, "diagnostic")
		exit, _ := strconv.Atoi(code)
		os.Exit(exit)
	}
	t.Parallel()
	l := &landing{Options: Options{Repo: t.TempDir()}}
	for _, code := range []string{"0", "3"} {
		output, err := l.run(t.Context(), append(os.Environ(), streamChild+"="+code), os.Args[0], "-test.run=^TestLandingRunKeepsEachStreamOwned$")
		if output != "result" {
			t.Errorf("exit %s: standard output = %q, want the result once", code, output)
		}
		if failed := code != "0"; failed != (err != nil) || failed && !strings.Contains(err.Error(), "diagnostic") {
			t.Errorf("exit %s: error = %v, want a failure to carry the diagnostic", code, err)
		}
	}
}

// writeModule writes a fixture module's files under repo.
func writeModule(t *testing.T, repo string, files map[string]string) {
	t.Helper()
	for name, content := range files {
		path := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestLaneReachSkipsOtherPackagesTests reaches a lane from production code
// in its closure and from its own packages' tests, never from another
// package's tests: a test file compiles only into its own test binary.
func TestLaneReachSkipsOtherPackagesTests(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	writeModule(t, repo, map[string]string{
		"go.mod":      "module fixture\n\ngo 1.26\n",
		"a/a.go":      "package a\n\nimport \"fixture/b\"\n\nvar A = b.B\n",
		"a/a_test.go": "package a\n",
		"b/b.go":      "package b\n\nvar B = 1\n",
		"b/b_test.go": "package b\n",
		"c/c.go":      "package c\n",
	})
	candidate := lane{packages: []string{"./a"}}
	reaches, err := discover(repo, candidate)
	if err != nil {
		t.Fatal(err)
	}
	for changed, want := range map[string]bool{
		"b/b.go":      true,
		"a/a_test.go": true,
		"b/b_test.go": false,
		"c/c.go":      false,
		// Data nested under a closure package, such as testdata or an
		// embedded recipe, reaches the lane; under another package it does
		// not.
		"b/testdata/golden.json": true,
		"c/testdata/golden.json": false,
		// A module file can move any lane's dependencies.
		"go.mod": true,
	} {
		if reached := reaches(candidate, []string{changed}); reached != want {
			t.Errorf("a change to %s reached the lane = %t, want %t", changed, reached, want)
		}
	}
}

// TestLaneSelectionUsesTheLaneBuildContext selects each tagged lane of the
// table in the build context its command runs in. In a fixture module the
// lane's package imports a dependency only from a test file behind the
// lane's tags, and the dependency's one file carries them too: a change to
// it selects the lane, but neither the same packages listed without the
// tags nor another lane sharing the tagged listing. Unrelated documentation
// selects nothing, and a dependency the tagged build cannot load refuses
// the selection while the untagged one, which never compiles it, still
// lists. No lane's command names tags of its own: the go command takes
// them from the descriptor.
func TestLaneSelectionUsesTheLaneBuildContext(t *testing.T) {
	t.Parallel()
	var tagged []lane
	for _, candidate := range lanes {
		if slices.Contains(candidate.command(""), "-tags") {
			t.Errorf("lane %s's command names its own tags; its build context belongs in the descriptor", candidate.name)
		}
		if candidate.tags != "" {
			tagged = append(tagged, candidate)
		}
	}
	if len(tagged) == 0 {
		t.Fatal("no lane builds with tags; the tagged suites would run nowhere")
	}
	for _, candidate := range tagged {
		t.Run(candidate.name, func(t *testing.T) {
			t.Parallel()
			constraint := "//go:build " + strings.ReplaceAll(candidate.tags, ",", " && ") + "\n\n"
			repo := t.TempDir()
			writeModule(t, repo, map[string]string{
				"go.mod":           "module fixture\n\ngo 1.26\n",
				"a/a.go":           "package a\n",
				"a/tagged_test.go": constraint + "package a\n\nimport _ \"fixture/b\"\n",
				"b/b.go":           constraint + "package b\n",
				"c/c.go":           "package c\n",
				"docs/notes.md":    "notes\n",
			})
			own := lane{name: candidate.name, packages: []string{"./a"}, tags: candidate.tags}
			untagged := lane{name: "untagged", packages: own.packages}
			neighbour := lane{name: "neighbour", packages: []string{"./c"}, tags: candidate.tags}
			reaches, err := discover(repo, own, untagged, neighbour)
			if err != nil {
				t.Fatal(err)
			}
			for _, selection := range []struct {
				selected lane
				changed  string
				want     bool
			}{
				{own, "b/b.go", true},
				{own, "a/tagged_test.go", true},
				{own, "c/c.go", false},
				{own, "docs/notes.md", false},
				{untagged, "b/b.go", false},
				{neighbour, "b/b.go", false},
				{neighbour, "c/c.go", true},
			} {
				if reached := reaches(selection.selected, []string{selection.changed}); reached != selection.want {
					t.Errorf("a change to %s selected lane %s = %t, want %t", selection.changed, selection.selected.name, reached, selection.want)
				}
			}
			writeModule(t, repo, map[string]string{"a/tagged_test.go": constraint + "package a\n\nimport _ \"fixture/missing\"\n"})
			if _, err := discover(repo, own); err == nil || !strings.Contains(err.Error(), "missing") {
				t.Errorf("a dependency the tagged build cannot load: discovery error = %v, want it named", err)
			}
			if _, err := discover(repo, untagged); err != nil {
				t.Errorf("the untagged build, which never compiles the tagged test, refused: %v", err)
			}
		})
	}
}

// TestPlanAndLoopEditsReachLanesOnlyThroughProduction selects lanes over
// the real module the way the lanes stage does, for an edit to the plan, to
// the loop driver and to the scorecard. A lane any of them reaches must
// compile the file into its production closure in its build context, the
// code its tests build and run; a test import alone would make every plan or
// loop edit pay for device work that never runs it. The cure is moving the
// fixture that needs the import out of the lane or importing the smaller
// owner it uses. The loop driver and the scorecard it measures with reach
// neither the audio nor the cuda lane at all: cmd/recipe, which both lanes
// build, imports neither, so a driver or scorecard edit runs no speech model
// and no CUDA suite.
func TestPlanAndLoopEditsReachLanesOnlyThroughProduction(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	reaches, err := discover(repo, lanes...)
	if err != nil {
		t.Fatal(err)
	}
	loopEdits := []string{"internal/loop/driver.go", "internal/scorecard/scorecard.go"}
	loopFree := []string{"audio", "cuda"}
	for _, name := range loopFree {
		if !slices.ContainsFunc(lanes, func(candidate lane) bool { return candidate.name == name }) {
			t.Fatalf("no lane is named %s; the loop-free check would hold nothing", name)
		}
	}
	for _, candidate := range lanes {
		t.Run(candidate.name, func(t *testing.T) {
			t.Parallel()
			production, err := gosource.HostBuildSelection(repo, append([]string{"-tags", candidate.tags, "-deps"}, candidate.packages...)...)
			if err != nil {
				t.Fatal(err)
			}
			unreachable := slices.Contains(loopFree, candidate.name)
			for _, edit := range append([]string{"internal/plan/plan.go"}, loopEdits...) {
				reached := reaches(candidate, []string{edit})
				if reached && !production.Files[edit] {
					t.Errorf("an edit to %s reaches lane %s only through a test import", edit, candidate.name)
				}
				if reached && unreachable && slices.Contains(loopEdits, edit) {
					t.Errorf("an edit to %s reaches lane %s; the loop driver and scorecard must stay out of it", edit, candidate.name)
				}
			}
		})
	}
}

// TestTrainingKernelEditReachesOnlyTrainingLanes selects lanes over the real
// module the way the lanes stage does, for an edit to a trainer-only device
// kernel and to its test. devicegrad's importers are trainers that no
// serving package builds, so the edit runs the cuda lane, which builds the
// trainers, and no other: not the device lane's kernel contracts, the
// served-model lanes (webui, model-journey, e4b) or audio. The same kernel
// in devicemath reached six lanes, through the serving packages that also
// train (trainingworkflow's dense-causal backend, modelbuilder's scratch
// trainer, mediacapability's diffusion VJPs).
func TestTrainingKernelEditReachesOnlyTrainingLanes(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	const kernel = "internal/devicegrad/windowed_attention_backward_cuda_windows.go"
	edits := []string{kernel, strings.TrimSuffix(kernel, ".go") + "_test.go"}
	for _, edit := range edits {
		if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(edit))); err != nil {
			t.Fatalf("the trainer-only kernel is not where the lanes are asked about: %v", err)
		}
	}
	reaches, err := discover(repo, lanes...)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range lanes {
		t.Run(candidate.name, func(t *testing.T) {
			t.Parallel()
			for _, edit := range edits {
				if reached, want := reaches(candidate, []string{edit}), candidate.name == "cuda"; reached != want {
					t.Errorf("an edit to %s reaches lane %s = %t, want %t", edit, candidate.name, reached, want)
				}
			}
		})
	}
}

// TestCUDATestsNeedALane holds the rules stage to the CUDA-gated tests'
// runner: a package whose tests call cudatest.Require is named unless a
// device lane scope covers it or a lane that opts into the device runs it,
// once however many of its tests call it. Production source that quotes the
// call is not a test and names nothing. The rule walks a fixture tree here,
// so this package's tests read none of the repository's.
func TestCUDATestsNeedALane(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	gated := "package fixture\n\nfunc TestDevice(t *testing.T) { " + cudaRequirement + "t) }\n"
	for name, content := range map[string]string{
		"internal/unlaned/a_test.go":      gated,
		"internal/unlaned/b_test.go":      gated,
		"internal/inference/c_test.go":    gated,
		"internal/cuda/kernels/d_test.go": gated,
		"internal/quoting/e.go":           gated,
		"cmd/hostonly/f_test.go":          "package main\n",
	} {
		file := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(file, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	uncovered, err := uncoveredCUDATests(repo)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(uncovered, []string{"internal/unlaned"}) {
		t.Fatalf("uncovered CUDA tests = %q, want only internal/unlaned", uncovered)
	}
}

// TestLanesMeasurementLine pins the lanes stage's entry in the Measured line
// over fake lane results: the discovery time, the turn wait, each passed
// lane with the admission waits its device lines add up to and the packages
// its go test lines executed or replayed from cache, a lane whose output
// carries neither with its time alone, and the unreached lanes as skipped.
// With no lane reached there is no turn to wait for.
func TestLanesMeasurementLine(t *testing.T) {
	t.Parallel()
	device := "[device] phase=1/3 heartbeat=running elapsed=0.0s command=go run ./cmd/cuda-info\n" +
		"[device] resource=GPU-1 mode=shared state=admitted wait=1.5s\n" +
		"[device] resource=GPU-1 mode=shared state=admitted wait=250ms\n"
	cuda := "ok  \tovergo/internal/hybridtrain\t61.204s\n" +
		"ok  \tovergo/internal/torchrng\t(cached)\n" +
		"ok  \tovergo/cmd/recipe\t(cached)\n" +
		"    trainer_test.go:12: ok  \tquoted by a test log\n" +
		"?   \tovergo/internal/fixture\t[no test files]\n"
	got := describeLanes(41200*time.Millisecond, 12*time.Minute, []string{
		describeLane("device", 2*time.Minute, device),
		describeLane("cuda", 30*time.Minute+time.Second, cuda),
		describeLane("webui", 3*time.Second, "webui lane: PASS browser=edge\n"),
	}, []string{"audio", "e4b"})
	want := "discovery 41.2s, turn wait 12m0s, device passed 2m0s admission 1.75s, " +
		"cuda passed 30m1s executed 1 cached 2, webui passed 3s, skipped audio e4b"
	if got != want {
		t.Errorf("lanes measurement = %q, want %q", got, want)
	}
	if got := describeLanes(900*time.Millisecond, 0, nil, []string{"device", "cuda"}); got != "discovery 900ms, skipped device cuda" {
		t.Errorf("lanes measurement with no lane reached = %q", got)
	}
	// The device lane writes the admitted line; reworded, it would read as a
	// lane that never claimed the device.
	source, err := os.ReadFile(filepath.Join("..", "..", "cmd", "device-lane", "admission.go"))
	if err != nil || !strings.Contains(string(source), admittedMarker) {
		t.Fatalf("cmd/device-lane no longer prints %q when the device admits a step: %v", admittedMarker, err)
	}
}

// TestAudioLaneOptsIntoTheStoreAcceptance holds the audio acceptances to a
// runner: go test skips them without the store-acceptance opt-in, so the lane
// that reaches internal/audioparity must set it.
func TestAudioLaneOptsIntoTheStoreAcceptance(t *testing.T) {
	t.Parallel()
	index := slices.IndexFunc(lanes, func(candidate lane) bool { return slices.Contains(candidate.packages, "./internal/audioparity") })
	if index < 0 {
		t.Fatal("no lane reaches internal/audioparity; its acceptances would run nowhere")
	}
	if !slices.Contains(lanes[index].environment, testskip.StoreAcceptanceEnv+"=1") {
		t.Fatalf("lane %s runs internal/audioparity without %s=1; its acceptances would skip", lanes[index].name, testskip.StoreAcceptanceEnv)
	}
}
