package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/processcontrol"
	"overgo/internal/testutil"
)

// Correctness commands coexist with a foreign shared holder. Admission precedes
// the single execution; command failures retain their own disposition.
func TestDeviceLaneAdmissionWaitsWithinBudget(t *testing.T) {
	t.Run("exclusive_owner_prevents_launch", func(t *testing.T) {
		name := "device-lane-wait/" + t.TempDir()
		command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^$")
		command.Env = append(os.Environ(), admissionProbeEnvironment+"="+name, "OVERGO_DEVICE_LANE_EXCLUSIVE_HOLD=1")
		input, err := command.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		output, err := command.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := command.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = input.Close(); _ = command.Process.Kill(); _ = command.Wait() })
		if line, err := bufio.NewReader(output).ReadString('\n'); err != nil || strings.TrimSpace(line) != "probe: admitted" {
			t.Fatalf("exclusive owner readiness: %q, %v", line, err)
		}
		runs := 0
		_, err = runStepAdmitted(t.Context(), &bytes.Buffer{}, name, 50*time.Millisecond, func(context.Context) (int, error) {
			runs++
			return 0, nil
		})
		if runs != 0 || !errors.Is(err, errAdmissionBudget) || !errors.Is(err, processcontrol.ErrResourceBusy) {
			t.Fatalf("waiting launched %d commands, error %v; want zero launches and admission expiry", runs, err)
		}
		if err := input.Close(); err != nil {
			t.Fatal(err)
		}
		if err := command.Wait(); err != nil {
			t.Fatal(err)
		}
		if _, err := runStepAdmitted(t.Context(), &bytes.Buffer{}, name, deviceAdmissionBudget, func(context.Context) (int, error) { runs++; return 0, nil }); err != nil || runs != 1 {
			t.Fatalf("released owner: %d launches, %v", runs, err)
		}
	})
	name := "device-lane-command/" + filepath.Base(t.TempDir())
	release, err := processcontrol.ShareResource(name)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	ctx, cancel := context.WithTimeoutCause(t.Context(), 5*time.Second, errors.New("shared command probe did not terminate"))
	defer cancel()
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=^$")
	command.Env = append(os.Environ(), admissionProbeEnvironment+"="+name, "OVERGO_DEVICE_LANE_SHARED_STEP=1")
	result, err := command.CombinedOutput()
	if err != nil || !strings.Contains(string(result), "probe: admitted") || !strings.Contains(string(result), "mode=shared state=admitted") {
		t.Fatalf("correctness command alongside a foreign shared holder: %v\n%s", err, result)
	}
	var output bytes.Buffer
	if code, err := runStepAdmitted(t.Context(), &output, name, deviceAdmissionBudget, func(context.Context) (int, error) { return 1, nil }); err != nil || code != 1 {
		t.Fatalf("failed step = code %d, %v, want its own exit code at once", code, err)
	}
	launch := errors.New("launch failed")
	if _, err := runStepAdmitted(t.Context(), &output, name, deviceAdmissionBudget, func(context.Context) (int, error) { return 0, launch }); !errors.Is(err, launch) {
		t.Fatalf("launch failure = %v, want it returned at once", err)
	}
}

// Admission cannot shorten execution or turn a command result into a retry.
func TestDeviceLaneAdmissionSeparatesWaitingFromExecution(t *testing.T) {
	var output bytes.Buffer
	budget := 50 * time.Millisecond
	code, err := runStepAdmitted(t.Context(), &output, "GPU-test", budget, func(ctx context.Context) (int, error) {
		select {
		case <-time.After(4 * budget):
			return 0, nil
		case <-ctx.Done():
			return 0, context.Cause(ctx)
		}
	})
	if code != 0 || err != nil {
		t.Fatalf("admitted step outliving the budget = code %d, %v, want completion", code, err)
	}
	errCaller := errors.New("the caller's bound")
	bounded, cancel := context.WithTimeoutCause(t.Context(), 4*budget, errCaller)
	defer cancel()
	began := time.Now()
	code, err = runStepAdmitted(bounded, &output, "GPU-test", budget, func(ctx context.Context) (int, error) {
		<-ctx.Done()
		return 0, context.Cause(ctx)
	})
	if code != 0 || !errors.Is(err, errCaller) || errors.Is(err, errAdmissionBudget) {
		t.Fatalf("stuck step = code %d, %v, want the caller's cause alone", code, err)
	}
	if elapsed := time.Since(began); elapsed < 3*budget || elapsed > 5*time.Second {
		t.Fatalf("stuck step ended after %s, want the caller's bound", elapsed)
	}
	runs := 0
	code, err = runStepAdmitted(t.Context(), &output, "GPU-test", budget, func(context.Context) (int, error) {
		runs++
		return processcontrol.ResourceBusyExitCode, nil
	})
	if code != processcontrol.ResourceBusyExitCode || err != nil || runs != 1 {
		t.Fatalf("command returned code %d, %v, executions %d; want original status after one execution", code, err, runs)
	}
}

// TestDeviceLaneTestStepsHoldSharedLease pins the lease around a test step:
// admission is waited for under the budget and reported like the gate's
// batch lease, the release reports the device not busy, an exhausted budget
// names the contention and its cause, and any other failure returns at once.
func TestDeviceLaneTestStepsHoldSharedLease(t *testing.T) {
	var output bytes.Buffer
	refusals, released := 0, 0
	share := func() (func() error, error) {
		if refusals < 2 {
			refusals++
			return nil, processcontrol.ErrResourceBusy
		}
		return func() error { released++; return nil }, nil
	}
	release, err := holdSharedLease(t.Context(), &output, "GPU-test", deviceAdmissionBudget, share)
	if err != nil {
		t.Fatal(err)
	}
	if err := release(); err != nil || released != 1 || refusals != 2 {
		t.Fatalf("release = %v, released %d, refusals %d", err, released, refusals)
	}
	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) != 3 || !strings.Contains(lines[0], "mode=shared state=waiting budget=") ||
		!strings.Contains(lines[1], "mode=shared state=admitted wait=") || !strings.Contains(lines[2], "mode=shared state=not_busy scope=step") {
		t.Fatalf("lease report = %q", output.String())
	}
	output.Reset()
	if _, err := holdSharedLease(t.Context(), &output, "GPU-test", 120*time.Millisecond, func() (func() error, error) { return nil, processcontrol.ErrResourceBusy }); !errors.Is(err, processcontrol.ErrResourceBusy) || !errors.Is(err, errAdmissionBudget) {
		t.Fatalf("exhausted budget = %v, want the contention and the budget cause", err)
	}
	if strings.Contains(output.String(), "state=admitted") {
		t.Fatal("an exhausted budget reported admission")
	}
	other := errors.New("lease failed")
	if _, err := holdSharedLease(t.Context(), &output, "GPU-test", deviceAdmissionBudget, func() (func() error, error) { return nil, other }); !errors.Is(err, other) {
		t.Fatalf("non-contention failure = %v, want it returned at once", err)
	}
}

const admissionProbeEnvironment = "OVERGO_DEVICE_LANE_ADMISSION_PROBE"

// Child probes exercise shared command admission and exclusive measurement
// admission against a foreign holder. Other runs execute the test suite.
func TestMain(m *testing.M) {
	name := os.Getenv(admissionProbeEnvironment)
	if name == "" {
		os.Exit(m.Run())
	}
	if os.Getenv("OVERGO_DEVICE_LANE_SHARED_STEP") != "" {
		code, err := runStepAdmitted(context.Background(), os.Stdout, name, 50*time.Millisecond, func(ctx context.Context) (int, error) {
			receipt, err := processcontrol.Run(ctx, processcontrol.Command{
				Path: os.Args[0], Args: []string{"-test.run=^$"},
				Env:    append(os.Environ(), "OVERGO_DEVICE_LANE_SHARED_STEP=", "OVERGO_DEVICE_LANE_COMMAND_REQUIRED=1"),
				Stdout: os.Stdout, Stderr: os.Stderr,
			})
			return receipt.ExitCode, err
		})
		if err != nil || code != 0 {
			fmt.Println("probe:", code, err)
			os.Exit(1)
		}
		fmt.Println("probe: admitted")
		return
	}
	var err error
	if os.Getenv("OVERGO_DEVICE_LANE_COMMAND_REQUIRED") != "" {
		var release func() error
		release, err = processcontrol.ShareResource(name)
		if err == nil {
			defer release()
		}
	} else {
		err = processcontrol.ClaimResource(name)
	}
	switch {
	case err == nil:
		fmt.Println("probe: admitted")
	case errors.Is(err, processcontrol.ErrResourceBusy):
		fmt.Println("probe: busy")
		if os.Getenv("OVERGO_DEVICE_LANE_COMMAND_REQUIRED") != "" {
			os.Exit(processcontrol.ResourceBusyExitCode)
		}
	default:
		fmt.Println("probe:", err)
		os.Exit(1)
	}
	if err == nil && os.Getenv("OVERGO_DEVICE_LANE_EXCLUSIVE_HOLD") != "" {
		_, _ = io.Copy(io.Discard, os.Stdin)
	}
}

// TestDeviceLaneMeasurementRunsOutsideSharedLease pins the parent/child
// interaction with a synthetic resource and no device: while the lane holds
// the shared lease a descendant's exclusive claim is refused, so a
// measurement child beneath the lease would wait on its own ancestor; once
// the lease is released the same claim is admitted. The step split keeps
// every measurement test out of the correctness run and runs only those
// tests in the measurement run, so no measurement child starts beneath the
// lease.
func TestDeviceLaneMeasurementRunsOutsideSharedLease(t *testing.T) {
	name := "device-lane-test/" + filepath.Base(t.TempDir())
	probe := func() string {
		command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^$", "-test.timeout=30s")
		command.Env = append(os.Environ(), admissionProbeEnvironment+"="+name)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("probe: %v: %s", err, output)
		}
		for _, verdict := range []string{"probe: admitted", "probe: busy"} {
			if strings.Contains(string(output), verdict) {
				return verdict
			}
		}
		t.Fatalf("probe reported nothing: %s", output)
		return ""
	}
	release, err := holdSharedLease(t.Context(), &bytes.Buffer{}, name, deviceAdmissionBudget, func() (func() error, error) { return processcontrol.ShareResource(name) })
	if err != nil {
		t.Fatal(err)
	}
	if verdict := probe(); verdict != "probe: busy" {
		t.Fatalf("exclusive claim beneath the lane's shared lease = %q, want contention", verdict)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
	if verdict := probe(); verdict != "probe: admitted" {
		t.Fatalf("exclusive claim after the lease = %q, want admission", verdict)
	}

	root := t.TempDir()
	directory := filepath.Join(root, "pkg")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	source := "package pkg\n\nimport \"testing\"\n\nfunc TestCorrect(t *testing.T) {}\n\n" +
		"func TestMeasured(t *testing.T) {\n\tif cudatest.MeasurementProcess(t, 0) {\n\t\treturn\n\t}\n}\n\n" +
		"func TestDeviceMeasured(t *testing.T) {\n\tif cudatest.MeasurementProcess(t, 0) {\n\t\treturn\n\t}\n}\n"
	if err := os.WriteFile(filepath.Join(directory, "pkg_test.go"), []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	step := deviceTestStep("./pkg")
	names, err := measurementTests(root, step)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(names, []string{"TestDeviceMeasured", "TestMeasured"}) {
		t.Fatalf("measurement tests = %v", names)
	}
	correctness, measurement := splitTestStep(step, names)
	if !slices.Equal(correctness, append(slices.Clone(step), "-skip", "^(TestDeviceMeasured|TestMeasured)$")) {
		t.Fatalf("correctness run = %v", correctness)
	}
	if !slices.Equal(measurement, append(slices.Clone(step), "-run", "^(TestDeviceMeasured|TestMeasured)$")) {
		t.Fatalf("measurement run = %v", measurement)
	}
	selected := deviceTestStep("-run", "Device", "./pkg")
	correctness, measurement = splitTestStep(selected, names)
	if !slices.Equal(correctness, append(slices.Clone(selected), "-skip", "^(TestDeviceMeasured)$")) || !slices.Equal(measurement, append(slices.Clone(selected), "-run", "^(TestDeviceMeasured)$")) {
		t.Fatalf("selected split = %v / %v", correctness, measurement)
	}
	plain := deviceTestStep("./internal/cuda/kernel")
	if correctness, measurement := splitTestStep(plain, nil); !slices.Equal(correctness, plain) || measurement != nil {
		t.Fatalf("step without measurement tests = %v / %v", correctness, measurement)
	}
}

// TestDeviceLaneStepsExitWithTheirOwnStatus pins the step form the admission
// relies on: a `go run` step becomes a built binary, so its exit status is
// the command's own, while a `go test` step keeps its form.
func TestDeviceLaneStepsExitWithTheirOwnStatus(t *testing.T) {
	buildDir := t.TempDir()
	test := deviceTestStep("./internal/cuda/kernel")
	if command, err := executableStep(t.Context(), testutil.RepoRoot(t), buildDir, test); err != nil || !slices.Equal(command, test) {
		t.Fatalf("go test step = %v, %v; want it unchanged", command, err)
	}
	command, err := executableStep(t.Context(), testutil.RepoRoot(t), buildDir, []string{"go", "run", "./cmd/cuda-info"})
	if err != nil {
		t.Fatal(err)
	}
	if len(command) != 1 || !strings.HasPrefix(command[0], buildDir) {
		t.Fatalf("go run step = %v, want one binary under %s", command, buildDir)
	}
	if _, err := os.Stat(command[0]); err != nil {
		t.Fatal(err)
	}
}
