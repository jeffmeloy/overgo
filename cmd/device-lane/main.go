// device-lane: the CUDA/device verification lane (floor component 8; the
// unique remainder of the retired verify.ps1). Runs the env-gated CUDA
// integration tests and the device probes with explicit reporting: a missing
// device or toolkit is UNAVAILABLE, never passing evidence.
//
// Full and manifest-scoped runs share declared device contracts.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"time"
	"unicode"

	"overgo/internal/clioptions"
	"overgo/internal/devicescope"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

const cudaTestEnv = "OVERGO_CUDA_TEST"

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
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	csv := *pathsFlag
	if *pathsFileFlag != "" {
		raw, err := os.ReadFile(*pathsFileFlag)
		if err != nil {
			return err
		}
		csv = string(raw)
	}
	paths := splitPaths(csv)
	plan, err := devicescope.PlanFor(".", paths)
	if err != nil {
		return err
	}
	steps := deviceSteps(plan)
	steps = append([][]string{{"go", "run", "./cmd/cuda-info"}}, steps...)
	buildDir, err := os.MkdirTemp("", "device-lane-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(buildDir)
	for index, step := range steps {
		began := time.Now()
		fmt.Println(deviceProgress(step, index, len(steps)))
		command, err := executableStep(ctx, ".", buildDir, step)
		if err != nil {
			return runrecord.LaneError(runrecord.LaneFailed, err.Error())
		}
		var stdout, stderr bytes.Buffer
		// Each step's contexts are created in the card's turn, which a landing takes once for its lanes, its processes
		// riding on the landing's place (card-handover/lane-turn).
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{
			Path: command[0], Args: command[1:], Env: append(os.Environ(), cudaTestEnv+"=1"),
			Stdout: &stdout, Stderr: &stderr,
		})
		fmt.Print(passedPackageLines(stdout.String()))
		if err == nil && receipt.ExitCode == 0 {
			err = verifyDeviceContracts(command, stdout.String())
		}
		out := stdout.String() + stderr.String()
		if err == nil && receipt.ExitCode != 0 {
			err = fmt.Errorf("exit code %d", receipt.ExitCode)
		}
		fmt.Printf("[device] %-60s %6.1fs %s\n", strings.Join(step[1:], " "), time.Since(began).Seconds(), clioptions.Verdict(err))
		if receipt.WallNS > 0 {
			fmt.Printf("[device] phase=%d/%d resource=not_busy scope=task process_tree=exited\n", index+1, len(steps))
		}
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
	// Render every failed package's bounded diagnostics from the step's test
	// JSON so an early package failure is not hidden behind a later passing
	// package's output. The shared streaming parser drops each passing tail and
	// keeps only failed ones, so retention stays bounded. Output that is not
	// test JSON (a launch or build failure) falls back to the bounded tail.
	report, parseErr := testevidence.GoTestJSONReader(strings.NewReader(out), false, clioptions.DiagnosticTailBytes, nil)
	if parseErr != nil || len(report.Failed) == 0 && len(report.Diagnostics) == 0 {
		return clioptions.Tail(out, clioptions.DiagnosticTailBytes)
	}
	var builder strings.Builder
	if len(report.Failed) != 0 {
		fmt.Fprintf(&builder, "[device] failed: %s\n", strings.Join(report.Failed, ", "))
	}
	for _, diagnostic := range report.Diagnostics {
		builder.WriteString(diagnostic)
		if !strings.HasSuffix(diagnostic, "\n") {
			builder.WriteByte('\n')
		}
	}
	return builder.String()
}

// deviceProgress is the line a step says as it starts, by its count: the lane's progress is its own count of steps, where
// a heartbeat every 15 s said only that time passed (card-handover/counts).
func deviceProgress(step []string, index, total int) string {
	return fmt.Sprintf("[device] step %d of %d: %s", index+1, total, strings.Join(step, " "))
}

func deviceSteps(plan devicescope.VerificationPlan) [][]string {
	steps := [][]string{{"go", "run", "./cmd/cuda-smoke"}}
	var packages []string
	var scoped [][]string
	for _, scope := range devicescope.Scopes(plan) {
		pattern := scope.Run
		if len(scope.Tests) > 0 {
			var names []string
			for _, name := range scope.Tests {
				names = append(names, regexp.QuoteMeta(name))
			}
			pattern = "^(" + strings.Join(names, "|") + ")$"
		}
		if pattern == "" {
			packages = append(packages, scope.Package)
		} else {
			scoped = append(scoped, deviceTestStep("-run", pattern, scope.Package))
		}
	}
	if len(packages) > 0 {
		steps = append(steps, deviceTestStep(packages...))
	}
	return append(steps, scoped...)
}

// deviceTestStep is the go test command of the device tests arguments select, each run afresh: go test's cache keys
// no file outside the module, so a cached pass would outlive a driver update or a changed model under the data root.
// No clock ends it: a stalled suite is diagnosed from its own state, with the lane's cancellation and process cleanup
// (adaptiveparity alone measured 624 s).
func deviceTestStep(arguments ...string) []string {
	command := []string{"go", "test", "-json", "-p=1", "-timeout=0"}
	command = append(command, arguments...)
	return append(command, freshRun)
}

// freshRun is go test's flag that runs a test again whatever its cache holds.
const freshRun = "-count=1"

// passedPackageLines is go test's line for each package that passed in stream, a go test -json stream, so the landing
// reads from the lane's output how many packages ran and how many go test replayed from its cache.
func passedPackageLines(stream string) string {
	var lines strings.Builder
	for line := range strings.Lines(stream) {
		var event struct{ Action, Output string }
		if json.Unmarshal([]byte(line), &event) == nil && event.Action == "output" && strings.HasPrefix(event.Output, "ok  \t") {
			lines.WriteString(event.Output)
		}
	}
	return lines.String()
}
