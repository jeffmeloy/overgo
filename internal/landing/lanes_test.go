package landing

import (
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

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

// TestLaneReachSkipsOtherPackagesTests reaches a lane from production code
// in its closure and from its own packages' tests, never from another
// package's tests: a test file compiles only into its own test binary.
func TestLaneReachSkipsOtherPackagesTests(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	for name, content := range map[string]string{
		"go.mod":      "module fixture\n\ngo 1.26\n",
		"a/a.go":      "package a\n\nimport \"fixture/b\"\n\nvar A = b.B\n",
		"a/a_test.go": "package a\n",
		"b/b.go":      "package b\n\nvar B = 1\n",
		"b/b_test.go": "package b\n",
		"c/c.go":      "package c\n",
	} {
		path := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	candidate := lane{packages: []string{"./a"}}
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
	} {
		l := &landing{Options: Options{Repo: repo, Log: io.Discard}, paths: []string{changed}}
		reached, err := l.reaches(t.Context(), candidate)
		if err != nil {
			t.Fatal(err)
		}
		if reached != want {
			t.Errorf("a change to %s reached the lane = %t, want %t", changed, reached, want)
		}
	}
}

// TestPlanAndLoopEditsReachLanesOnlyThroughProduction selects lanes over
// the real module the way the lanes stage does, for an edit to the plan and
// one to the loop. A lane either edit reaches must hold the package in its
// production closure, the code its tests build and run; a test import alone
// would make every plan or loop edit pay for device work that never runs
// it. The cure is moving the fixture that needs the import out of the lane
// or importing the smaller owner it uses. The loop's one real reach today is
// cmd/recipe's production import of it (the breaker, rollback, promotion and
// re-entry verbs), which the audio and cuda lanes build.
func TestPlanAndLoopEditsReachLanesOnlyThroughProduction(t *testing.T) {
	t.Parallel()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range lanes {
		t.Run(candidate.name, func(t *testing.T) {
			t.Parallel()
			l := &landing{Options: Options{Repo: repo, Log: io.Discard}}
			production, err := l.listDirectories(t.Context(), append([]string{"-deps"}, candidate.packages...))
			if err != nil {
				t.Fatal(err)
			}
			for _, edit := range []string{"internal/plan/plan.go", "internal/loop/driver.go"} {
				l.paths = []string{edit}
				reached, err := l.reaches(t.Context(), candidate)
				if err != nil {
					t.Fatal(err)
				}
				if owner := path.Join(filepath.ToSlash(repo), path.Dir(edit)); reached && !production[owner] {
					t.Errorf("an edit to %s reaches lane %s only through a test import", edit, candidate.name)
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
