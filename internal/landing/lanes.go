package landing

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/devicescope"
	"overgo/internal/gosource"
	"overgo/internal/processcontrol"
	"overgo/internal/testskip"
)

// lane is one check that drives real hardware or a browser. It runs when the
// change reaches the test closure of its packages, or touches a path under
// one of its prefixes; go list, not a manifest, decides the closure.
type lane struct {
	name     string
	packages []string
	prefixes []string
	// tags are the lane's build context: go lists its closure and runs its
	// command, after the command's verb, with them.
	tags    string
	command func(changed string) []string
	// environment is added to the gate's own for the lane's command.
	environment []string
}

var lanes = []lane{
	{
		// The device lane runs its scopes' tests as its own processes, not as
		// imports, so its reach is the closure of those packages.
		name: "device", packages: append(devicePackages(), "./cmd/device-lane"), prefixes: []string{"kernels/"},
		command: func(changed string) []string { return []string{"run", "./cmd/device-lane", "-paths-file", changed} },
	},
	{
		// The page's assets, browser tests, lane and page declarations.
		name: "webui", packages: []string{"./internal/server", "./internal/webuilane", "./cmd/webui-lane"},
		prefixes: []string{"internal/server/webui/", "internal/server/webui_", "cmd/webui-lane/", "internal/webuilane/",
			"internal/server/workspace_manifest.json", "internal/server/workspace_schema.json"},
		command: func(string) []string { return []string{"run", "./cmd/webui-lane"} },
	},
	{
		// A served-model journey or the server command.
		name:     "model-journey",
		packages: []string{"./internal/modelswap", "./internal/libraryintake", "./internal/discovery", "./cmd/server"},
		prefixes: []string{"cmd/server/", "internal/server/webui_browser_firstrun", "internal/server/webui_browser_library_validation",
			"internal/server/webui_browser_preparation", "internal/server/webui_browser_store_test.go", "internal/server/webui_served_projector",
			// The microphone journey lives in audioparity with the fixtures
			// it shares, so any file there reaches it.
			"internal/audioparity/"},
		command: func(string) []string { return []string{"run", "./cmd/webui-lane", "-journeys"} },
	},
	{
		// The audio acceptances run the recorded speech models against the
		// reference store; the landing's go test skips them.
		// cmd/evaluate and cmd/recipe are the producers the acceptances
		// build and run as their own processes.
		name: "audio", packages: []string{"./internal/audioparity", "./cmd/evaluate", "./cmd/recipe"},
		command:     func(string) []string { return []string{"test", "./internal/audioparity"} },
		environment: []string{testskip.StoreAcceptanceEnv + "=1"},
	},
	{
		// Thoughtbank's real-checkpoint suite, behind the modeltest tag:
		// training parity, code completion and the process probe on the
		// local Fractale checkpoint.
		name: "thoughtbank", packages: []string{"./internal/thoughtbank"}, tags: "modeltest",
		command:     func(string) []string { return []string{"test", "./internal/thoughtbank"} },
		environment: []string{cudaTestEnvironment},
	},
	{
		// The integration-tagged suites: Wan's real-model runtime and
		// reference edit on the local Wan2.1-T2V-1.3B and the retained
		// capture beside the data, and the quantizer's pinned checks.
		name: "integration", packages: []string{"./internal/latentvideo", "./internal/quant"}, tags: "integration",
		command:     func(string) []string { return []string{"test", "./internal/latentvideo", "./internal/quant"} },
		environment: []string{cudaTestEnvironment},
	},
	{
		// E4B served over every HTTP route with text, image and audio input.
		// The package's resource-recovery proof (about 220 s) is an
		// evidence producer, run when its evidence is refreshed.
		name: "e4b", packages: []string{"./internal/server"},
		command:     func(string) []string { return []string{"test", "./internal/server", "-run", "^TestE4BHTTPModalities$"} },
		environment: []string{cudaTestEnvironment},
	},
	{
		// The CUDA suites the device lane's fixture-free scopes leave out:
		// the real-model inference and dense-causal checks, image and video
		// generation, media capability, the trainers and the device kernels
		// only they call, the scratch and routed models, adaptive parity,
		// the composite-generation parity on Wan, long-form evaluation and
		// the recipe, evaluate and train commands. One package at a time
		// shares the device; go test's cache replays each until code it
		// reads changes.
		name: "cuda", packages: cudaPackages,
		// adaptiveparity alone runs about 11 minutes, past go test's
		// default per-binary timeout.
		command:     func(string) []string { return append([]string{"test", "-p=1", "-timeout=0"}, cudaPackages...) },
		environment: []string{cudaTestEnvironment},
	},
}

// cudaTestEnvironment opts a lane's tests into the real device; without it
// cudatest.Require skips them.
const cudaTestEnvironment = "OVERGO_CUDA_TEST=1"

// cudaRequirement is the call a real-device test opens with, split so that
// source quoting it is not read as calling it.
const cudaRequirement = "cudatest" + ".Require("

// uncoveredCUDATests names each package under internal/ or cmd/ whose tests
// call cudatest.Require but that no device lane scope covers and no lane
// runs with cudaTestEnvironment: a real-device suite lands with its runner,
// never behind a variable nothing sets. It walks the tree at every landing;
// as a test it would make the directory listings a cache input of this
// package and rerun it on every edit anywhere.
func uncoveredCUDATests(repo string) ([]string, error) {
	var uncovered []string
	for _, top := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(repo, top), func(name string, entry fs.DirEntry, err error) error {
			// A repository without the top directory holds no tests there.
			if errors.Is(err, fs.ErrNotExist) && name == filepath.Join(repo, top) {
				return fs.SkipDir
			}
			if err != nil || entry.IsDir() || !strings.HasSuffix(name, "_test.go") {
				return err
			}
			data, err := os.ReadFile(name)
			if err != nil || !bytes.Contains(data, []byte(cudaRequirement)) {
				return err
			}
			relative, err := filepath.Rel(repo, filepath.Dir(name))
			if directory := filepath.ToSlash(relative); err == nil && !cudaCovered(directory) && !slices.Contains(uncovered, directory) {
				uncovered = append(uncovered, directory)
			}
			return err
		})
		if err != nil {
			return nil, err
		}
	}
	return uncovered, nil
}

// cudaCovered reports whether a device lane scope covers the package
// directory or a lane runs it with the device opted in.
func cudaCovered(directory string) bool {
	return lane{packages: devicePackages()}.owns(directory) || slices.ContainsFunc(lanes, func(candidate lane) bool {
		return slices.Contains(candidate.environment, cudaTestEnvironment) && candidate.owns(directory)
	})
}

// cudaPackages are the packages whose CUDA tests the cuda lane runs.
var cudaPackages = []string{
	"./internal/diffusionimage", "./internal/latentimage", "./internal/mediacapability", "./internal/scratchmodel", "./internal/scratchtrain",
	"./internal/modeldevice", "./internal/hybridtrain", "./internal/e4btrain", "./internal/devicegrad", "./internal/densecausaltrain", "./internal/controllertrain", "./internal/trainingprogram",
	"./internal/torchrng", "./internal/routedlm", "./internal/longform", "./internal/sensenovarecipe",
	"./internal/inference", "./internal/densecausal", "./internal/adaptiveparity", "./internal/composition",
	"./cmd/evaluate", "./cmd/longform", "./cmd/recipe", "./cmd/train",
}

// laneTurn names the host-wide turn a landing holds while its lanes run.
const laneTurn = "overgo-landing-lanes"

// lanes runs every lane the change reaches, one after another: they share
// the device. They run in the host's lane turn, so another landing on this
// machine waits for it rather than running its lanes beside these on the
// one GPU. A turn, not a claim on the device: a claim would make every lane
// process its member and erase the process identities the device's own
// admission tests exercise. The measurement says how long finding the
// reached lanes and waiting for the turn took, and how each lane ended; a
// lane that refuses the landing logs its time.
func (l *landing) lanes(ctx context.Context) error {
	changed, err := os.CreateTemp("", "overgo-landing-paths-*.txt")
	if err != nil {
		return err
	}
	defer os.Remove(changed.Name())
	if _, err := changed.WriteString(strings.Join(l.paths, "\n")); err != nil {
		changed.Close()
		return err
	}
	if err := changed.Close(); err != nil {
		return err
	}
	var reached []lane
	var skipped []string
	started := time.Now()
	reaches, err := discover(l.Repo, lanes...)
	if err != nil {
		return err
	}
	for _, candidate := range lanes {
		if reaches(candidate, l.paths) {
			reached = append(reached, candidate)
		} else {
			skipped = append(skipped, candidate.name)
		}
	}
	discovery := time.Since(started).Round(time.Millisecond)
	if len(reached) == 0 {
		l.details["lanes"] = describeLanes(discovery, 0, nil, skipped)
		return nil
	}
	fmt.Fprintln(l.Log, "land: lanes wait for the host's lane turn")
	waiting := time.Now()
	return processcontrol.ResourceTransaction(laneTurn, func() error {
		turnWait := time.Since(waiting).Round(time.Millisecond)
		var passed []string
		for _, candidate := range reached {
			fmt.Fprintf(l.Log, "land: lane %s runs\n", candidate.name)
			var environment []string
			if len(candidate.environment) > 0 {
				environment = append(os.Environ(), candidate.environment...)
			}
			began := time.Now()
			output, err := l.run(ctx, environment, "go", slices.Insert(candidate.command(changed.Name()), 1, "-tags", candidate.tags)...)
			wall := time.Since(began).Round(time.Millisecond)
			if err != nil {
				fmt.Fprintf(l.Log, "land: lane %s refused after %s\n", candidate.name, wall)
				return fmt.Errorf("lane %s: %w", candidate.name, err)
			}
			passed = append(passed, describeLane(candidate.name, wall, output))
		}
		l.details["lanes"] = describeLanes(discovery, turnWait, passed, skipped)
		return nil
	})
}

// passedPackage opens go test's line for a package that passed, in its
// plain output and in the output events of -json alike.
const passedPackage = "ok  \t"

// admittedMarker precedes the wait in the line the device lane prints when
// the device admits one of its steps (cmd/device-lane's holdSharedLease).
const admittedMarker = "state=admitted wait="

// describeLane is a passed lane's entry in the measurement: its wall time;
// the device admission wait its admitted lines add up to, where the lane
// claims the device; and, where its output carries go test's package lines,
// how many packages go test executed and how many it replayed from cache.
func describeLane(name string, wall time.Duration, output string) string {
	claimed := false
	var admission time.Duration
	executed, cached := 0, 0
	for line := range strings.Lines(output) {
		_, wait, found := strings.Cut(line, admittedMarker)
		if waited, err := time.ParseDuration(strings.TrimSpace(wait)); found && err == nil {
			claimed, admission = true, admission+waited
		}
		switch {
		case !strings.HasPrefix(line, passedPackage):
		case strings.Contains(line, cachedMarker):
			cached++
		default:
			executed++
		}
	}
	entry := fmt.Sprintf("%s passed %s", name, wall)
	if claimed {
		entry += fmt.Sprintf(" admission %s", admission)
	}
	if executed+cached > 0 {
		entry += fmt.Sprintf(" executed %d cached %d", executed, cached)
	}
	return entry
}

// describeLanes is the lanes stage's measurement: the time finding the
// reached lanes took, the wait for the host's lane turn when a lane ran,
// each passed lane's entry, and the lanes the change did not reach.
func describeLanes(discovery, turnWait time.Duration, passed, skipped []string) string {
	entries := []string{"discovery " + discovery.String()}
	if len(passed) > 0 {
		entries = append(entries, "turn wait "+turnWait.String())
	}
	entries = append(entries, passed...)
	if len(skipped) > 0 {
		entries = append(entries, "skipped "+strings.Join(skipped, " "))
	}
	return strings.Join(entries, ", ")
}

// discover lists the lanes' test closures once per build context -- lanes
// share a listing only when they build with the same tags -- and returns
// whether changed paths reach a lane: a module file reaches every lane, a
// path under one of its prefixes reaches it, and so does a path in its test
// closure, what its own packages' test binaries import, or such a package
// imports without tests. A package or dependency a listing cannot load
// refuses the landing.
func discover(repo string, candidates ...lane) (func(lane, []string) bool, error) {
	patterns := map[string][]string{}
	for _, candidate := range candidates {
		for _, pattern := range candidate.packages {
			// A lane package this checkout does not hold, as in a fixture
			// repository, reaches nothing; one it holds must load.
			base, _ := strings.CutSuffix(strings.TrimPrefix(pattern, "./"), "/...")
			if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(base))); err == nil {
				patterns[candidate.tags] = append(patterns[candidate.tags], pattern)
			}
		}
	}
	selections := map[string]gosource.BuildSelection{}
	for tags, packages := range patterns {
		selection, err := gosource.HostBuildSelection(repo, append([]string{"-tags", tags, "-deps", "-test"}, packages...)...)
		if err != nil {
			return nil, err
		}
		selections[tags] = selection
	}
	return func(candidate lane, paths []string) bool {
		selection, directories, closure := selections[candidate.tags], map[string]string{}, map[string]bool{}
		for file, importPath := range selection.Packages {
			directories[importPath] = path.Dir(file)
		}
		// The listing holds every lane of the build context, so the lane's
		// closure is taken from its own packages: each one, what it imports,
		// and what its test binary imports. A dependency compiled for a test
		// binary is listed as "path [binary]"; one outside the repository has
		// no directory.
		for file, importPath := range selection.Packages {
			if !candidate.owns(path.Dir(file)) {
				continue
			}
			for _, dependency := range slices.Concat([]string{importPath}, selection.Dependencies[importPath], selection.Dependencies[importPath+".test"]) {
				dependency, _, _ = strings.Cut(dependency, " [")
				closure[directories[dependency]] = true
			}
		}
		return slices.ContainsFunc(paths, func(name string) bool {
			if name == "go.mod" || name == "go.sum" || slices.ContainsFunc(candidate.prefixes, func(prefix string) bool { return strings.HasPrefix(name, prefix) }) {
				return true
			}
			// A test file compiles only into its own package's test binary,
			// so it reaches the lane only when that package is the lane's.
			if strings.HasSuffix(name, "_test.go") {
				return candidate.owns(path.Dir(name))
			}
			// A data file nested under a closure package, testdata or an
			// embedded recipe, is that package's input; a Go file belongs to
			// its own package, so only data walks up.
			for parent := path.Dir(name); !closure[parent]; parent = path.Dir(parent) {
				if strings.HasSuffix(name, ".go") || parent == "." {
					return false
				}
			}
			return true
		})
	}, nil
}

// owns reports whether one of the lane's package patterns, ./directory or
// ./directory/..., names the directory.
func (candidate lane) owns(directory string) bool {
	return slices.ContainsFunc(candidate.packages, func(pattern string) bool {
		base, recursive := strings.CutSuffix(strings.TrimPrefix(pattern, "./"), "/...")
		return directory == base || recursive && strings.HasPrefix(directory, base+"/")
	})
}

// devicePackages are the device lane's scopes as package patterns; the cuda
// tree is recursive.
func devicePackages() []string {
	var patterns []string
	for _, packagePath := range devicescope.Packages() {
		if packagePath == "internal/cuda" {
			packagePath += "/..."
		}
		patterns = append(patterns, "./"+packagePath)
	}
	return patterns
}

func tail(text string) string {
	return clioptions.Tail(text, clioptions.DiagnosticTailBytes)
}
