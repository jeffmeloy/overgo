package landing

import (
	"context"
	"fmt"
	"os"
	"path"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/devicescope"
	"overgo/internal/testskip"
)

// lane is one check that drives real hardware or a browser. It runs when the
// change reaches the test closure of its packages, or touches a path under
// one of its prefixes; go list, not a manifest, decides the closure.
type lane struct {
	name     string
	packages []string
	prefixes []string
	command  func(changed string) []string
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
		name: "thoughtbank", packages: []string{"./internal/thoughtbank"},
		command:     func(string) []string { return []string{"test", "-tags", "modeltest", "./internal/thoughtbank"} },
		environment: []string{"OVERGO_CUDA_TEST=1"},
	},
	{
		// The integration-tagged suites: Wan's real-model runtime and
		// reference edit on the local Wan2.1-T2V-1.3B and the retained
		// capture beside the data, and the quantizer's pinned checks.
		name: "integration", packages: []string{"./internal/latentvideo", "./internal/quant"},
		command: func(string) []string {
			return []string{"test", "-tags", "integration", "./internal/latentvideo", "./internal/quant"}
		},
		environment: []string{"OVERGO_CUDA_TEST=1"},
	},
	{
		// E4B served over every HTTP route with text, image and audio input.
		// The package's resource-recovery proof (about 220 s) is an
		// evidence producer, run when its evidence is refreshed.
		name: "e4b", packages: []string{"./internal/server"},
		command:     func(string) []string { return []string{"test", "./internal/server", "-run", "^TestE4BHTTPModalities$"} },
		environment: []string{"OVERGO_CUDA_TEST=1"},
	},
	{
		// The CUDA suites the device lane's fixture-free scopes leave out:
		// the real-model inference and dense-causal checks, image and video
		// generation, media capability, the trainers, the scratch and routed
		// models, adaptive parity, long-form evaluation and the recipe,
		// evaluate and composite-generation commands. One package at a time
		// shares the device; go test's cache replays each until code it
		// reads changes.
		name: "cuda", packages: cudaPackages,
		// adaptiveparity alone runs about 11 minutes, past go test's
		// default per-binary timeout.
		command:     func(string) []string { return append([]string{"test", "-p=1", "-timeout=0"}, cudaPackages...) },
		environment: []string{"OVERGO_CUDA_TEST=1"},
	},
}

// cudaPackages are the packages whose CUDA tests the cuda lane runs.
var cudaPackages = []string{
	"./internal/diffusionimage", "./internal/latentimage", "./internal/mediacapability", "./internal/scratchmodel",
	"./internal/modeldevice", "./internal/hybridtrain", "./internal/controllertrain", "./internal/trainingprogram",
	"./internal/torchrng", "./internal/routedlm", "./internal/longform", "./internal/sensenovarecipe",
	"./internal/inference", "./internal/densecausal", "./internal/adaptiveparity",
	"./cmd/evaluate", "./cmd/longform", "./cmd/recipe", "./cmd/composite-generation-lane",
}

// lanes runs every lane the change reaches, one after another: they share
// the device.
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
	for _, candidate := range lanes {
		reached, err := l.reaches(ctx, candidate)
		if err != nil {
			return err
		}
		if !reached {
			continue
		}
		fmt.Fprintf(l.Log, "land: lane %s runs\n", candidate.name)
		var environment []string
		if len(candidate.environment) > 0 {
			environment = append(os.Environ(), candidate.environment...)
		}
		if _, err := l.run(ctx, environment, "go", candidate.command(changed.Name())...); err != nil {
			return fmt.Errorf("lane %s: %w", candidate.name, err)
		}
	}
	return nil
}

// reaches reports whether a changed path lies in a package of the lane's
// test closure, or under a prefix that triggers it. The module files reach
// every lane: a dependency change can move any of them.
func (l *landing) reaches(ctx context.Context, candidate lane) (bool, error) {
	if slices.ContainsFunc(l.paths, func(name string) bool {
		return name == "go.mod" || name == "go.sum" ||
			slices.ContainsFunc(candidate.prefixes, func(prefix string) bool { return strings.HasPrefix(name, prefix) })
	}) {
		return true, nil
	}
	closure, err := l.listDirectories(ctx, append([]string{"-deps", "-test"}, candidate.packages...))
	if err != nil {
		return false, err
	}
	own, err := l.listDirectories(ctx, candidate.packages)
	if err != nil {
		return false, err
	}
	root := strings.ReplaceAll(l.Repo, "\\", "/")
	for _, name := range l.paths {
		directory := path.Join(root, path.Dir(name))
		// A test file compiles only into its own package's test binary, so
		// it reaches the lane only when that package is the lane's.
		if strings.HasSuffix(name, "_test.go") {
			if own[directory] {
				return true, nil
			}
			continue
		}
		if closure[directory] {
			return true, nil
		}
		// A data file nested under a closure package, testdata or an
		// embedded recipe, is that package's input; a Go file belongs to its
		// own package, so only data walks up.
		for parent := directory; !strings.HasSuffix(name, ".go") && parent != root && parent != path.Dir(parent); {
			parent = path.Dir(parent)
			if closure[parent] {
				return true, nil
			}
		}
	}
	return false, nil
}

// listDirectories is the set of package directories go list names for the
// arguments, with forward slashes.
func (l *landing) listDirectories(ctx context.Context, arguments []string) (map[string]bool, error) {
	listed, err := l.run(ctx, nil, "go", append([]string{"list", "-e", "-f", "{{.Dir}}"}, arguments...)...)
	if err != nil {
		return nil, err
	}
	directories := map[string]bool{}
	for directory := range strings.Lines(listed) {
		directories[strings.ReplaceAll(strings.TrimSpace(directory), "\\", "/")] = true
	}
	return directories, nil
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
