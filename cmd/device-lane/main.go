// device-lane: the CUDA/device verification lane (floor component 8; the
// unique remainder of the retired verify.ps1). Runs the env-gated CUDA
// integration tests and the device probes with honest reporting: a missing
// device or toolkit is UNAVAILABLE, never passing evidence.
//
// Target form (Automation Doctrine Layer 3) scopes kernels by manifest diff;
// this faithful port runs the full device set and is the interim named by
// skill.md Testing Lanes.
package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
)

const cudaTestEnv = "OVERGO_CUDA_TEST"

// Two default Go test windows. Measured adaptiveparity device suite: 624.107s.
// Revisit when stored device-run history supports a tighter derived bound.
const devicePackageTimeout = "20m"

var pathsFlag = flag.String("paths", "", "comma-separated changed paths; empty runs the full device set")

func main() {
	flag.Parse()
	clioptions.MainNamed("device-lane", run)
}

func run() error {
	start := time.Now()
	// cuda-info first: it is the availability probe. Failure here means no
	// usable device/driver -- report UNAVAILABLE and exit nonzero so a caller
	// can never mistake absence for green.
	if out, err := clioptions.CombinedOutput(os.Environ(), "go", "run", "./cmd/cuda-info"); err != nil {
		fmt.Println("device-lane: UNAVAILABLE -- cuda-info failed; no passing evidence exists")
		fmt.Print(clioptions.Tail(out, 800))
		return runrecord.LaneError(runrecord.LaneUnavailable, err.Error())
	}
	paths := splitPaths(*pathsFlag)
	steps, err := deviceSteps(paths)
	if err != nil {
		return err
	}
	for _, step := range steps {
		began := time.Now()
		out, err := clioptions.CombinedOutput(append(os.Environ(), cudaTestEnv+"=1"), step[0], step[1:]...)
		fmt.Printf("[device] %-60s %6.1fs %s\n", strings.Join(step[1:], " "), time.Since(began).Seconds(), clioptions.Verdict(err))
		if err != nil {
			fmt.Print(clioptions.Tail(out, 2000))
			return runrecord.LaneError(runrecord.LaneFailed, strings.Join(step, " ")+" failed")
		}
	}
	fmt.Printf("=== DEVICE LANE GREEN in %.1fs ===\n", time.Since(start).Seconds())
	fmt.Printf("honesty: device scope=%s\n", deviceScopeLabel(paths))
	return nil
}

func splitPaths(csv string) []string {
	var paths []string
	for _, path := range strings.Split(csv, ",") {
		if path = strings.TrimSpace(strings.ReplaceAll(path, "\\", "/")); path != "" {
			paths = append(paths, path)
		}
	}
	return paths
}

func deviceSteps(paths []string) ([][]string, error) {
	steps := [][]string{{"go", "run", "./cmd/cuda-smoke"}}
	if len(paths) == 0 {
		return append(steps,
			deviceTestStep("./internal/cuda/...", "./internal/model", "./internal/projector", "./internal/optimizer", "./internal/devicemath"),
			deviceTestStep("-run", "Device", "./internal/densecausal")), nil
	}
	packages := map[string]bool{}
	tests := map[string]map[string]bool{}
	for _, path := range paths {
		parts := strings.Split(path, "/")
		switch {
		case strings.HasPrefix(path, "kernels/") || strings.HasPrefix(path, "internal/cuda/"):
			packages["./internal/cuda/..."] = true
		case len(parts) > 2 && parts[0] == "internal":
			pkg := "./internal/" + parts[1]
			if !strings.HasSuffix(path, "_test.go") {
				packages[pkg] = true
				continue
			}
			names, err := changedDeviceTests(path)
			if err != nil {
				return nil, err
			}
			if names == nil {
				packages[pkg] = true
				continue
			}
			if tests[pkg] == nil {
				tests[pkg] = map[string]bool{}
			}
			for _, name := range names {
				tests[pkg][name] = true
			}
		}
	}
	ordered := make([]string, 0, len(packages))
	for pkg := range packages {
		ordered = append(ordered, pkg)
	}
	sort.Strings(ordered)
	if len(ordered) > 0 {
		// One device owner: package concurrency invalidates wall and peak ratchets.
		steps = append(steps, deviceTestStep(ordered...))
	}
	var testPackages []string
	for pkg := range tests {
		if !packages[pkg] {
			testPackages = append(testPackages, pkg)
		}
	}
	sort.Strings(testPackages)
	for _, pkg := range testPackages {
		var names []string
		for name := range tests[pkg] {
			names = append(names, name)
		}
		sort.Strings(names)
		steps = append(steps, deviceTestStep("-run", "^("+strings.Join(names, "|")+")$", pkg))
	}
	return steps, nil
}

func changedDeviceTests(path string) ([]string, error) {
	file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.SkipObjectResolution)
	if err != nil {
		return nil, fmt.Errorf("device-lane: parse changed test %s: %w", path, err)
	}
	var names []string
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if ok && function.Recv == nil &&
			(strings.HasPrefix(function.Name.Name, "Test") || strings.HasPrefix(function.Name.Name, "Example")) {
			names = append(names, function.Name.Name)
		}
	}
	return names, nil
}

func deviceTestStep(arguments ...string) []string {
	command := []string{"go", "test", "-p=1", "-timeout=" + devicePackageTimeout}
	command = append(command, arguments...)
	return append(command, "-count=1")
}

func deviceScopeLabel(paths []string) string {
	if len(paths) == 0 {
		return "full"
	}
	return fmt.Sprintf("changed-paths(%d)", len(paths))
}
