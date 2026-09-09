// device-lane: the CUDA/device verification lane (floor component 8; the
// unique remainder of the retired verify.ps1). Runs the env-gated CUDA
// integration tests and the device probes with explicit reporting: a missing
// device or toolkit is UNAVAILABLE, never passing evidence.
//
// Target form (Automation Doctrine Layer 3) scopes kernels by manifest diff;
// this faithful port runs the full device set and is the interim named by
// skill.md Testing Lanes.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
	"unicode"

	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
)

const cudaTestEnv = "OVERGO_CUDA_TEST"

// Two default Go test windows. Measured adaptiveparity device suite: 624.107s.
// Revisit when stored device-run history supports a tighter derived bound.
const devicePackageTimeout = "20m"

var (
	pathsFlag     = flag.String("paths", "", "comma-separated changed paths; empty runs the full device set")
	pathsFileFlag = flag.String("paths-file", "", "file of newline-separated changed paths; overrides -paths (Windows caps the command line)")
)

func main() {
	flag.Parse()
	clioptions.MainNamed("device-lane", run)
}

func run() error {
	start := time.Now()
	csv := *pathsFlag
	if *pathsFileFlag != "" {
		raw, err := os.ReadFile(*pathsFileFlag)
		if err != nil {
			return err
		}
		csv = string(raw)
	}
	paths := splitPaths(csv)
	plan, err := automationcheck.DevicePlan(".", paths)
	if err != nil {
		return err
	}
	steps := deviceSteps(plan)
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
			fmt.Print(deviceDiagnostic(out, err))
			if index == 0 {
				return runrecord.LaneError(runrecord.LaneUnavailable, "cuda-info failed; no passing evidence exists")
			}
			return runrecord.LaneError(runrecord.LaneFailed, strings.Join(step, " ")+" failed")
		}
	}
	fmt.Printf("=== DEVICE LANE GREEN in %.1fs ===\n", time.Since(start).Seconds())
	if plan.Full {
		fmt.Printf("audit: device scope=full reason=%s\n", plan.Reason)
	} else {
		fmt.Printf("audit: device scope=packages(%d) functions(%d)\n", len(plan.Packages), len(plan.Functions))
	}
	return nil
}

func splitPaths(csv string) []string {
	return strings.FieldsFunc(strings.ReplaceAll(csv, "\\", "/"), func(r rune) bool { return r == ',' || unicode.IsSpace(r) })
}

// deviceDiagnostic renders a failed step's evidence: the diagnostic tail of
// what it printed, or, when the step printed nothing, the launch error
// itself, so a process that never started is named rather than mute.
func deviceDiagnostic(out string, err error) string {
	if strings.TrimSpace(out) == "" {
		return "[device] launch: " + err.Error() + "\n"
	}
	return clioptions.Tail(out, clioptions.DiagnosticTailBytes)
}

func deviceHeartbeat(step []string, index, total int, began time.Time, done <-chan struct{}, stopped chan<- struct{}) {
	defer close(stopped)
	ticker := time.Tick(runrecord.DefaultHeartbeatStaleAfter / 2)
	for {
		select {
		case <-done:
			return
		case <-ticker:
			fmt.Println(deviceProgress(step, index, total, time.Since(began)))
		}
	}
}

func deviceProgress(step []string, index, total int, elapsed time.Duration) string {
	return fmt.Sprintf("[device] phase=%d/%d heartbeat=running elapsed=%.1fs command=%s", index+1, total, elapsed.Seconds(), strings.Join(step, " "))
}

func deviceSteps(plan automationcheck.DeviceVerificationPlan) [][]string {
	steps := [][]string{{"go", "run", "./cmd/cuda-smoke"}}
	if plan.Full {
		// The full plan tests exactly the packages the lane declares it owns.
		full := []string{"./" + automationcheck.DeviceLanePackages[0] + "/..."}
		for _, packagePath := range automationcheck.DeviceLanePackages[1 : len(automationcheck.DeviceLanePackages)-1] {
			full = append(full, "./"+packagePath)
		}
		return append(steps,
			deviceTestStep(full...),
			deviceTestStep("-run", "Device", "./"+automationcheck.DeviceLanePackages[len(automationcheck.DeviceLanePackages)-1]))
	}
	if len(plan.Packages) > 0 {
		// Bound this lane's package concurrency; independent consumers share VRAM.
		steps = append(steps, deviceTestStep(plan.Packages...))
	}
	return steps
}

func deviceTestStep(arguments ...string) []string {
	command := []string{"go", "test", "-p=1", "-timeout=" + devicePackageTimeout}
	command = append(command, arguments...)
	return append(command, "-count=1")
}
