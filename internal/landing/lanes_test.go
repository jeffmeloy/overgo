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
// gate's and the cycle's tables in the build context its command runs in.
// In a fixture module the lane's package imports a dependency only from a
// test file behind the lane's tags, and the dependency's one file carries
// them too: a change to either selects the lane, but neither the same
// packages listed without the tags nor another lane sharing the tagged
// listing. Unrelated documentation selects nothing, and a dependency the
// tagged build cannot load refuses the selection while the untagged one,
// which never compiles it, still lists. No lane's command names tags of its
// own: the go command takes them from the descriptor.
func TestLaneSelectionUsesTheLaneBuildContext(t *testing.T) {
	t.Parallel()
	var tagged []lane
	for _, candidate := range slices.Concat(lanes, cycleLanes) {
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
				{untagged, "a/tagged_test.go", false},
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

// The lane sets reach rows name, each written once. Every lane a set names
// must be in the gate's or the cycle's table, so a renamed or removed lane
// fails the table rather than silently emptying the rows that name its set.
var (
	// servingLanes build or run what serves a model: the page at the gate,
	// and in the cycle the served-model journeys and the speech
	// acceptances.
	servingLanes = []string{"webui", "model-journey", "audio"}
	// browserOnlyLanes drive a browser and never the device. Every other
	// gate lane is a device lane, so a lane added to the gate's table counts
	// as one until it is named here.
	browserOnlyLanes = []string{"webui"}
)

// TestReachInvariants pins the boundaries between a changed path and the
// lanes it must never reach, and between a moved suite and the cycle lane
// that runs it, one row per invariant, all selected by one discovery over
// each real lane table, the way the lanes stage and validate -run select
// them. A row states what must never come back, a set of lanes the path
// stays out of, never the lanes it reaches today, so a lane it may
// legitimately reach later is no failure. A production row also holds every
// lane the path does reach to compiling it into that lane's production
// closure in the lane's build context, the code its tests build and run: a
// test import alone would make every such edit pay for a lane that never
// runs it, and the cure is moving the fixture that needs the import out of
// the lane or importing the smaller owner it uses. A row naming a cycle lane
// holds a suite moved off the gate: an edit to it must reach that lane, or
// the cycle would never run what the gate gave up. No gate lane builds with
// the cycle's tags, so a landing compiles none of the tests behind them.
//
// To pin a new boundary, add a row, never a test function: a representative
// path that exists, the named set it must not reach (add a set beside the
// others when none fits), whether only production code may carry it into a
// lane, the cycle lane that runs it when it is a moved suite, and the
// reason in one line.
func TestReachInvariants(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	everyLane := slices.Concat(lanes, cycleLanes)
	for _, name := range slices.Concat(servingLanes, browserOnlyLanes) {
		if !slices.ContainsFunc(everyLane, func(candidate lane) bool { return candidate.name == name }) {
			t.Fatalf("no lane is named %s; the rows naming its set would hold nothing", name)
		}
	}
	var gateLanes, deviceLanes []string
	for _, candidate := range lanes {
		if slices.ContainsFunc(strings.Split(candidate.tags, ","), func(tag string) bool { return slices.Contains(strings.Split(CycleTags, ","), tag) }) {
			t.Errorf("gate lane %s builds with the cycle's tags %s; a landing would run the suites the cycle owns", candidate.name, candidate.tags)
		}
		gateLanes = append(gateLanes, candidate.name)
		if !slices.Contains(browserOnlyLanes, candidate.name) {
			deviceLanes = append(deviceLanes, candidate.name)
		}
	}
	rows := []struct {
		path         string
		mustNotReach []string
		production   bool
		cycle        string
		reason       string
	}{
		{"internal/plan/plan.go", deviceLanes, true, "", "the plan is the loop's work list; editing it runs no model or kernel"},
		{"internal/landing/landing.go", deviceLanes, true, "", "the gate chooses the lanes; no lane runs the gate"},
		{"internal/loop/driver.go", deviceLanes, true, "", "the loop driver dispatches rows; no speech model or CUDA suite runs it"},
		{"internal/scorecard/scorecard.go", deviceLanes, true, "", "the scorecard reads recorded evidence; it measures nothing on the device"},
		{"internal/devicegrad/windowed_attention_backward_cuda_windows.go", servingLanes, false, "", "devicegrad holds kernels only trainers call, and no serving package links a trainer"},
		{"internal/devicegrad/windowed_attention_backward_cuda_windows_test.go", servingLanes, false, "", "a trainer kernel's test compiles only into devicegrad's own test binary"},
		{"internal/densecausaltrain/train_resident_cuda_windows.go", servingLanes, false, "", "the device dense-causal trainer is imported only by the commands that train"},
		{"internal/scratchtrain/resident_cuda_windows.go", servingLanes, false, "", "the device scratch trainer is imported only by trainers, never by the model builder the server links"},
		{"internal/e4btrain/model.go", servingLanes, false, "", "E4B training is a trainer; serving E4B never links it"},
		{"internal/devicemath/cuda_scope_windows.go", servingLanes, false, "", "no serving path calls devicemath; the served-model lanes link only host training"},
		{"internal/audioparity/vad_acceptance_test.go", gateLanes, false, "audio", "the speech acceptances serve the recorded models against the reference store"},
		{"cmd/server/main.go", gateLanes, false, "model-journey", "the served-model journeys build the server command and serve real models"},
		{"internal/server/webui_browser_firstrun_test.go", deviceLanes, false, "model-journey", "the first-run journey serves real models through the swap proxy"},
		{"internal/server/e4b_modalities_windows_test.go", deviceLanes, false, "models", "E4B over every HTTP route serves the real model"},
		{"internal/server/webui_library_projector_test.go", deviceLanes, false, "models", "the library projector hashes a stored model and projector pair"},
		{"internal/thoughtbank/load_model_test.go", gateLanes, false, "models", "the real-checkpoint suite opens Fractale beside the data"},
		{"internal/latentvideo/runtime_cuda_windows_test.go", gateLanes, false, "models", "Wan's runtime and reference edit open Wan2.1-T2V-1.3B beside the data"},
		{"internal/quant/quantize_weighted_windows_test.go", gateLanes, false, "models", "the quantizer's pinned check reads the reference build beside the data"},
		{"internal/diffusionimage/resident_sampling_cuda_windows_test.go", gateLanes, false, "models", "every device test of diffusionimage samples the real SimpleDiffusion checkpoint"},
		{"internal/mediacapability/video_cuda_windows_test.go", gateLanes, false, "models", "media capability activates real Wan, LiveEdit, Krea and RxBrain checkpoints"},
		{"internal/mediacapability/image_embedding_test.go", gateLanes, false, "models", "the image embedding capability loads the real DINO checkpoint"},
		{"internal/sensenovarecipe/runtime_cuda_windows_test.go", gateLanes, false, "models", "SenseNova generates from the real 8B checkpoint"},
		{"internal/sensenovarecipe/activate_model_test.go", gateLanes, false, "models", "SenseNova's activation derives from the real 8B checkpoint"},
		{"internal/adaptiveparity/qwen35_training_cuda_windows_test.go", gateLanes, false, "models", "adaptive parity trains real checkpoints beside the data"},
		{"internal/composition/composite_generation_parity_windows_test.go", gateLanes, false, "models", "composite generation runs the real Wan checkpoint"},
		{"internal/inference/runner_integration_test.go", gateLanes, false, "models", "the native oracles open real GGUFs"},
		{"internal/inference/qwen35_9b_serving_smoke_test.go", gateLanes, false, "models", "the 9B serving smoke loads the real Qwen3.5-9B"},
		{"internal/latentimage/krea_checkpoint_cuda_windows_model_test.go", gateLanes, false, "models", "the resident Krea checks open the real 12.8B checkpoint"},
		{"internal/latentimage/krea_checkpoint_model_test.go", gateLanes, false, "models", "the host Krea checks decode and encode with the real checkpoint"},
		{"internal/routedlm/flowterminal_model_test.go", gateLanes, false, "models", "the flow terminal runs the real SenseNova artifact"},
		{"internal/speechsynth/train_step_model_test.go", gateLanes, false, "models", "joint Muon training steps the real speech artifact"},
		{"internal/tabularicl/predict_wall_test.go", gateLanes, false, "models", "the predict wall loads every real TabICL head"},
		{"internal/modelintake/projection_test.go", gateLanes, false, "models", "projection intake hashes the stored model and projector pair"},
		{"cmd/train/route_tabular_tabfm_test.go", gateLanes, false, "models", "the TabFM objective trains the real classification head"},
		{"cmd/validate/accepted_coverage_report_test.go", gateLanes, false, "models", "the coverage projection reads the live store's accepted evidence"},
		{"cmd/validate/accepted_modalities_model_test.go", gateLanes, false, "models", "the modality coverage reads the live store's accepted evidence"},
		{"cmd/evaluate/transcription_projected_windows_test.go", gateLanes, false, "models", "projected transcription serves the real E4B"},
		{"cmd/longform/e4b_recovery_windows_test.go", gateLanes, false, "models", "long-form recovery serves the real E4B at 32768 tokens"},
		{"cmd/recipe/projection_execution_windows_test.go", gateLanes, false, "models", "projection execution serves the real E4B"},
	}
	gateReaches, err := discover(repo, lanes...)
	if err != nil {
		t.Fatal(err)
	}
	cycleReaches, err := discover(repo, cycleLanes...)
	if err != nil {
		t.Fatal(err)
	}
	reaches := func(candidate lane, paths []string) bool {
		if slices.ContainsFunc(lanes, func(gate lane) bool { return gate.name == candidate.name }) {
			return gateReaches(candidate, paths)
		}
		return cycleReaches(candidate, paths)
	}
	productions := map[string]gosource.BuildSelection{}
	for _, row := range rows {
		if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(row.path))); err != nil {
			t.Errorf("row %s holds nothing; point it at a path that still carries the boundary: %v", row.path, err)
			continue
		}
		if moved := slices.IndexFunc(cycleLanes, func(candidate lane) bool { return candidate.name == row.cycle }); row.cycle != "" && (moved < 0 || !cycleReaches(cycleLanes[moved], []string{row.path})) {
			t.Errorf("an edit to %s reaches no cycle lane %s, so the cycle never runs what the gate gave up: %s", row.path, row.cycle, row.reason)
		}
		for _, candidate := range everyLane {
			switch {
			case !reaches(candidate, []string{row.path}):
			case slices.Contains(row.mustNotReach, candidate.name):
				t.Errorf("an edit to %s reaches lane %s: %s", row.path, candidate.name, row.reason)
			case row.production:
				production, listed := productions[candidate.name]
				if !listed {
					production, err = gosource.HostBuildSelection(repo, append([]string{"-tags", candidate.tags, "-deps"}, candidate.packages...)...)
					if err != nil {
						t.Fatal(err)
					}
					productions[candidate.name] = production
				}
				if !production.Files[row.path] {
					t.Errorf("an edit to %s reaches lane %s only through a test import; move the fixture that needs it out of the lane or import the smaller owner it uses", row.path, candidate.name)
				}
			}
		}
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
// runner: go test skips them without the store-acceptance opt-in, so the
// cycle lane that reaches internal/audioparity must set it.
func TestAudioLaneOptsIntoTheStoreAcceptance(t *testing.T) {
	t.Parallel()
	index := slices.IndexFunc(cycleLanes, func(candidate lane) bool { return slices.Contains(candidate.packages, "./internal/audioparity") })
	if index < 0 {
		t.Fatal("no cycle lane reaches internal/audioparity; its acceptances would run nowhere")
	}
	if !slices.Contains(cycleLanes[index].environment, testskip.StoreAcceptanceEnv+"=1") {
		t.Fatalf("lane %s runs internal/audioparity without %s=1; its acceptances would skip", cycleLanes[index].name, testskip.StoreAcceptanceEnv)
	}
}
