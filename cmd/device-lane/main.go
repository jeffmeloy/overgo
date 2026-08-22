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
	"maps"
	"os"
	"slices"
	"strings"
	"time"
	"unicode"

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
	paths := splitPaths(*pathsFlag)
	steps := deviceSteps(paths)
	steps = append([][]string{{"go", "run", "./cmd/cuda-info"}}, steps...)
	for index, step := range steps {
		began := time.Now()
		done, stopped := make(chan struct{}), make(chan struct{})
		fmt.Println(deviceProgress(step, index, len(steps), 0))
		go deviceHeartbeat(step, index, len(steps), began, done, stopped)
		out, err := clioptions.CombinedOutput(append(os.Environ(), cudaTestEnv+"=1"), step[0], step[1:]...)
		close(done)
		<-stopped
		fmt.Printf("[device] %-60s %6.1fs %s\n", strings.Join(step[1:], " "), time.Since(began).Seconds(), clioptions.Verdict(err))
		if err != nil {
			fmt.Print(clioptions.Tail(out, 2000))
			if index == 0 {
				return runrecord.LaneError(runrecord.LaneUnavailable, "cuda-info failed; no passing evidence exists")
			}
			return runrecord.LaneError(runrecord.LaneFailed, strings.Join(step, " ")+" failed")
		}
	}
	fmt.Printf("=== DEVICE LANE GREEN in %.1fs ===\n", time.Since(start).Seconds())
	if len(paths) == 0 {
		fmt.Println("honesty: device scope=full")
	} else {
		fmt.Printf("honesty: device scope=changed-paths(%d)\n", len(paths))
	}
	return nil
}

func splitPaths(csv string) []string {
	return strings.FieldsFunc(strings.ReplaceAll(csv, "\\", "/"), func(r rune) bool { return r == ',' || unicode.IsSpace(r) })
}

func deviceHeartbeat(step []string, index, total int, began time.Time, done <-chan struct{}, stopped chan<- struct{}) {
	defer close(stopped)
	ticker := time.NewTicker(runrecord.DefaultHeartbeatStaleAfter / 2)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			fmt.Println(deviceProgress(step, index, total, time.Since(began)))
		}
	}
}

func deviceProgress(step []string, index, total int, elapsed time.Duration) string {
	return fmt.Sprintf("[device] phase=%d/%d heartbeat=running elapsed=%.1fs command=%s", index+1, total, elapsed.Seconds(), strings.Join(step, " "))
}

func deviceSteps(paths []string) [][]string {
	steps := [][]string{{"go", "run", "./cmd/cuda-smoke"}}
	if len(paths) == 0 {
		return append(steps,
			deviceTestStep("./internal/cuda/...", "./internal/model", "./internal/projector", "./internal/optimizer", "./internal/devicemath"),
			deviceTestStep("-run", "Device", "./internal/densecausal"))
	}
	packages := map[string]bool{}
	for _, path := range paths {
		parts := strings.Split(path, "/")
		switch {
		case strings.HasPrefix(path, "kernels/") || strings.HasPrefix(path, "internal/cuda/"):
			packages["./internal/cuda/..."] = true
		case len(parts) > 2 && parts[0] == "internal":
			packages["./internal/"+parts[1]] = true
		}
	}
	ordered := slices.Sorted(maps.Keys(packages))
	if len(ordered) > 0 {
		// One device owner: package concurrency invalidates wall and peak ratchets.
		steps = append(steps, deviceTestStep(ordered...))
	}
	return steps
}

func deviceTestStep(arguments ...string) []string {
	command := []string{"go", "test", "-p=1", "-timeout=" + devicePackageTimeout}
	command = append(command, arguments...)
	return append(command, "-count=1")
}
