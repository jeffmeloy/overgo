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
	"flag"
	"fmt"
	"os"
	"os/signal"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode"

	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
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
	plan, err := automationcheck.DevicePlan(".", paths)
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
	identity, err := deviceIdentity()
	if err != nil {
		return runrecord.LaneError(runrecord.LaneUnavailable, err.Error())
	}
	for index, step := range steps {
		began := time.Now()
		done, stopped := make(chan struct{}), make(chan struct{})
		fmt.Println(deviceProgress(step, index, len(steps), 0))
		go deviceHeartbeat(step, index, len(steps), began, done, stopped)
		command, err := executableStep(ctx, ".", buildDir, step)
		if err != nil {
			close(done)
			<-stopped
			return runrecord.LaneError(runrecord.LaneFailed, err.Error())
		}
		var stdout, stderr bytes.Buffer
		var receipt processcontrol.Receipt
		// Commands and correctness tests share admission. Measurement tests
		// run outside that lease so their exclusive children can acquire.
		run := func(ctx context.Context, command []string) (int, error) {
			stdout.Reset()
			stderr.Reset()
			var err error
			receipt, err = processcontrol.Run(ctx, processcontrol.Command{
				Path: command[0], Args: command[1:], Env: append(os.Environ(), cudaTestEnv+"=1"),
				Stdout: &stdout, Stderr: &stderr,
			})
			if err == nil && receipt.ExitCode == 0 {
				err = verifyDeviceContracts(command, stdout.String())
			}
			return receipt.ExitCode, err
		}
		var code int
		if index == 0 {
			// Device metadata creates no context and needs no GPU lease.
			code, err = run(ctx, command)
		} else if slices.Equal(command, step) {
			code, err = runTestStep(ctx, os.Stdout, identity, step, run)
		} else {
			code, err = runStepAdmitted(ctx, os.Stdout, identity, deviceAdmissionBudget, func(ctx context.Context) (int, error) { return run(ctx, command) })
		}
		out := stdout.String() + stderr.String()
		if err == nil && code != 0 {
			err = fmt.Errorf("exit code %d", code)
		}
		close(done)
		<-stopped
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
	var packages []string
	var scoped [][]string
	for _, scope := range automationcheck.DeviceScopes(plan) {
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

func deviceTestStep(arguments ...string) []string {
	command := []string{"go", "test", "-json", "-p=1", "-timeout=" + devicePackageTimeout}
	command = append(command, arguments...)
	return append(command, "-count=1")
}
