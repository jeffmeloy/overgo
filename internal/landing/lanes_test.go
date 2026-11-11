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

	"overgo/internal/testcheck"
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
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	}
}

// TestLaneReachSkipsOtherPackagesTests holds a lane's reach to its closure, and
// to naming what pulled the lane in: a path in its own package or under one of
// its prefixes is named alone, production code its closure takes in is named
// with the lane's package that took it in, and of several paths the first that
// reaches it is named. A test file reaches it only from its own packages, never
// another package's, since a test file compiles only into its own test binary;
// data nested under a closure package reaches it, and a module file can move
// any lane's dependencies.
func TestLaneReachSkipsOtherPackagesTests(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	writeModule(t, repo, map[string]string{
		"go.mod":      "module fixture\n\ngo 1.26\n",
		"a/a.go":      "package a\n\nimport \"fixture/b\"\n\nvar A = b.B\n",
		"a/a_test.go": "package a\n",
		"b/b.go":      "package b\n\nimport \"fixture/c\"\n\nvar B = c.C\n",
		"b/b_test.go": "package b\n",
		"c/c.go":      "package c\n\nvar C = 1\n",
		"d/d.go":      "package d\n",
	})
	candidate := lane{packages: []string{"./a"}, prefixes: []string{"web/"}}
	reaches := testcheck.Must(discover(repo, candidate))(t)
	for _, change := range []struct {
		paths []string
		want  string
	}{
		{[]string{"a/a.go"}, "a/a.go"},
		{[]string{"a/a_test.go"}, "a/a_test.go"},
		{[]string{"b/b.go"}, "b/b.go through a"},
		{[]string{"c/c.go"}, "c/c.go through a"},
		{[]string{"b/b_test.go"}, ""},
		{[]string{"d/d.go"}, ""},
		{[]string{"web/page.js"}, "web/page.js"},
		// Data nested under a closure package, such as testdata or an
		// embedded recipe, reaches the lane; under another package it does
		// not.
		{[]string{"b/testdata/golden.json"}, "b/testdata/golden.json through a"},
		{[]string{"d/testdata/golden.json"}, ""},
		{[]string{"go.mod"}, "go.mod"},
		{[]string{"d/d.go", "b/b.go", "c/c.go"}, "b/b.go through a"},
	} {
		if got := reaches(candidate, change.paths).by; got != change.want {
			t.Errorf("a change to %v reached the lane for %q, want %q", change.paths, got, change.want)
		}
	}
}

// TestLaneSelectionUsesTheLaneBuildContext selects lanes in the build
// context their commands run in, once for each context the tagged lanes of
// the gate's and the cycle's tables declare. In a fixture module the lane's
// package imports a dependency only from a test file behind the lane's tags,
// and the dependency's one file carries them too: a change to either selects
// the lane, but neither the same packages listed without the tags nor
// another lane sharing the tagged listing. Unrelated documentation selects
// nothing, and a dependency the tagged build cannot load refuses the
// selection while the untagged one, which never compiles it, still lists. No
// lane's command names tags of its own: the go command takes them from the
// descriptor.
func TestLaneSelectionUsesTheLaneBuildContext(t *testing.T) {
	t.Parallel()
	var tagged []string
	for _, candidate := range slices.Concat(lanes, cycleLanes) {
		if slices.Contains(candidate.command(""), "-tags") {
			t.Errorf("lane %s's command names its own tags; its build context belongs in the descriptor", candidate.name)
		}
		if candidate.tags != "" && !slices.Contains(tagged, candidate.tags) {
			tagged = append(tagged, candidate.tags)
		}
	}
	if len(tagged) == 0 {
		t.Fatal("no lane builds with tags; the tagged suites would run nowhere")
	}
	for _, tags := range tagged {
		t.Run(tags, func(t *testing.T) {
			t.Parallel()
			constraint := "//go:build " + strings.ReplaceAll(tags, ",", " && ") + "\n\n"
			repo := t.TempDir()
			writeModule(t, repo, map[string]string{
				"go.mod":           "module fixture\n\ngo 1.26\n",
				"a/a.go":           "package a\n",
				"a/tagged_test.go": constraint + "package a\n\nimport _ \"fixture/b\"\n",
				"b/b.go":           constraint + "package b\n",
				"c/c.go":           "package c\n",
				"docs/notes.md":    "notes\n",
			})
			own := lane{name: "tagged", packages: []string{"./a"}, tags: tags}
			untagged := lane{name: "untagged", packages: own.packages}
			neighbour := lane{name: "neighbour", packages: []string{"./c"}, tags: tags}
			reaches := testcheck.Must(discover(repo, own, untagged, neighbour))(t)
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
				if reached := reaches(selection.selected, []string{selection.changed}).by != ""; reached != selection.want {
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
	writeModule(t, repo, map[string]string{
		"internal/unlaned/a_test.go":      gated,
		"internal/unlaned/b_test.go":      gated,
		"internal/inference/c_test.go":    gated,
		"internal/cuda/kernels/d_test.go": gated,
		"internal/quoting/e.go":           gated,
		"cmd/hostonly/f_test.go":          "package main\n",
	})
	uncovered := testcheck.Must(uncoveredCUDATests(repo))(t)
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
// TestCPULanesOverlapTheDeviceLane holds the lanes a change reaches to the schedule that shares the
// device and no more: the lanes that need none start at once, beside the rest, and the lanes that
// share it keep their order and run one at a time; the page's lane is the one that needs no
// device, and the device and CUDA lanes are the ones that do.
func TestCPULanesOverlapTheDeviceLane(t *testing.T) {
	t.Parallel()
	names := func(candidates []lane) []string {
		var named []string
		for _, candidate := range candidates {
			named = append(named, candidate.name)
		}
		return named
	}
	beside, shared := splitLanes(lanes)
	if !slices.Equal(names(beside), []string{"webui"}) || !slices.Equal(names(shared), []string{"device", "cuda"}) {
		t.Fatalf("lanes split into %v beside and %v sharing the device, want the page beside the device and CUDA lanes in order", names(beside), names(shared))
	}
	reached := []lane{{name: "a"}, {name: "page", hostOnly: true}, {name: "b"}, {name: "other", hostOnly: true}}
	beside, shared = splitLanes(reached)
	if !slices.Equal(names(beside), []string{"page", "other"}) || !slices.Equal(names(shared), []string{"a", "b"}) {
		t.Fatalf("a host-only lane stayed in the device queue: %v beside, %v sharing", names(beside), names(shared))
	}
	for _, candidate := range cycleLanes {
		if candidate.hostOnly {
			t.Errorf("cycle lane %s is host-only: the cycle's lanes read real models and the store, and take their turns alone", candidate.name)
		}
	}
}

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
	var passed []string
	var admission time.Duration
	for _, described := range []struct {
		name   string
		wall   time.Duration
		output string
	}{{"device", 2 * time.Minute, device}, {"cuda", 30*time.Minute + time.Second, cuda}, {"webui", 3 * time.Second, "webui lane: PASS browser=edge\n"}} {
		entry, waited := describeLane(described.name, described.wall, described.output)
		passed, admission = append(passed, entry), admission+waited
	}
	if admission != 1750*time.Millisecond {
		t.Errorf("the lanes waited %s for the device, want the device lane's two admissions added", admission)
	}
	got := describeLanes(41200*time.Millisecond, 12*time.Minute, passed, []string{"audio", "e4b"})
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
	if !slices.Contains(cycleLanes[index].environment, cudaTestEnvironment) || cycleLanes[index].tags != CycleTags || !slices.Contains(cycleLanes[index].command(""), "-count=1") {
		t.Fatal("audio cycle must build and freshly execute the resident device reference")
	}
}

// TestARunnersImportsRunOnlyItsOwnTests holds the device and browser lanes' reach to their tests: each runner's own
// files reach its lane, through a prefix of its directory, and what the runner alone imports does not, since that
// cannot change a device or browser result and the runner's tests hold the commands it builds. A fixture module pins the
// reading: a lane whose runner imports a package is reached by the runner's own file and not by that package.
func TestARunnersImportsRunOnlyItsOwnTests(t *testing.T) {
	t.Parallel()
	for name, runner := range map[string]string{"device": "cmd/device-lane", "webui": "cmd/webui-lane"} {
		index := slices.IndexFunc(lanes, func(candidate lane) bool { return candidate.name == name })
		if index < 0 {
			t.Fatalf("no %s lane", name)
		}
		if candidate := lanes[index]; slices.Contains(candidate.packages, "./"+runner) || !slices.Contains(candidate.prefixes, runner+"/") {
			t.Errorf("the %s lane reaches through %v and prefixes %v, want its runner %s by its own files alone", name, candidate.packages, candidate.prefixes, runner)
		}
	}
	repo := t.TempDir()
	writeModule(t, repo, map[string]string{
		"go.mod":             "module fixture\n\ngo 1.26\n",
		"tests/tests.go":     "package tests\n",
		"runner/main.go":     "package main\n\nimport \"fixture/options\"\n\nfunc main() { _ = options.Name }\n",
		"options/options.go": "package options\n\nvar Name = \"lane\"\n",
	})
	candidate := lane{packages: []string{"./tests"}, prefixes: []string{"runner/"}}
	reaches := testcheck.Must(discover(repo, candidate))(t)
	if got := reaches(candidate, []string{"runner/main.go"}).by; got != "runner/main.go" {
		t.Errorf("a change to the runner's own file reached the lane for %q, want it", got)
	}
	if got := reaches(candidate, []string{"options/options.go"}).by; got != "" {
		t.Errorf("a change to what the runner alone imports reached the lane for %q, want none", got)
	}
}
