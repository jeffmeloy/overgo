//go:build windows

package device

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/processcontrol"
)

const contextAdmissionProbeEnvironment = "OVERGO_TEST_CONTEXT_ADMISSION"

func TestCUDAContextSharedAdmission(t *testing.T) {
	cudatest.Require(t)
	if role := os.Getenv(contextAdmissionProbeEnvironment); role != "" {
		runContextAdmissionConsumer(t, role)
		return
	}
	start := func(role string) (*exec.Cmd, io.WriteCloser, *bufio.Reader) {
		t.Helper()
		command := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestCUDAContextSharedAdmission$", "-test.timeout=1m")
		command.Env = append(os.Environ(), contextAdmissionProbeEnvironment+"="+role)
		input, err := command.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		output, err := command.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		command.Stderr = os.Stderr
		if err := command.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = input.Close(); _ = command.Process.Kill(); _ = command.Wait() })
		return command, input, bufio.NewReader(output)
	}
	expect := func(output *bufio.Reader, want string) {
		t.Helper()
		line, err := output.ReadString('\n')
		if err != nil || strings.TrimSpace(line) != want {
			t.Fatalf("CUDA process: got %q, want %q: %v", line, want, err)
		}
	}
	probe := func(role string) {
		t.Helper()
		command, input, output := start(role)
		expect(output, "verified")
		_ = input.Close()
		if _, err := io.Copy(io.Discard, output); err != nil {
			t.Fatal(err)
		}
		if err := command.Wait(); err != nil {
			t.Fatal(err)
		}
	}
	one, inputOne, outputOne := start("shared")
	expect(outputOne, "admitted")
	two, inputTwo, outputTwo := start("shared")
	expect(outputTwo, "admitted")
	probe("exclusive-busy")
	if err := one.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := one.Wait(); err == nil {
		t.Fatal("CUDA consumer death was not injected")
	}
	_ = inputOne.Close()
	if _, err := io.WriteString(inputTwo, "check\n"); err != nil {
		t.Fatal(err)
	}
	expect(outputTwo, "checked")
	probe("exclusive-busy")
	if _, err := io.WriteString(inputTwo, "release\n"); err != nil {
		t.Fatal(err)
	}
	expect(outputTwo, "released")
	probe("shared-probe")
	// Other interactive users may remain. Positive exclusive admission is proved
	// on isolated resource identities by processcontrol's lifecycle tests.
	_ = inputTwo.Close()
	if _, err := io.Copy(io.Discard, outputTwo); err != nil {
		t.Fatal(err)
	}
	if err := two.Wait(); err != nil {
		t.Fatal(err)
	}
	t.Log("two independent CUDA contexts allocate and execute; physical capacity refusal; process-local byte accounting; survivor remains correct after peer death; context closure and shared readmission; explicit measurement contention")
}

func runContextAdmissionConsumer(t *testing.T, role string) {
	t.Helper()
	if strings.HasPrefix(role, "exclusive-") {
		library, err := driver.Open()
		if err != nil {
			t.Fatal(err)
		}
		defer library.Close()
		_, err = library.ReserveDevice(0)
		if role == "exclusive-busy" {
			if !errors.Is(err, processcontrol.ErrResourceBusy) {
				t.Fatalf("wanted typed GPU contention: %v", err)
			}
		} else if err != nil {
			t.Fatal(err)
		}
		fmt.Println("verified")
		return
	}
	worker, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	values := []float32{3.5, 3.5}
	var pointer driver.DevicePtr
	err = worker.Do(t.Context(), func(state *State) error {
		var err error
		pointer, err = state.Driver.MemAlloc(uint64(len(driver.Bytes(values))))
		if err != nil {
			return err
		}
		_, total, err := state.Driver.MemInfo()
		if err != nil {
			return err
		}
		oversized, err := state.Driver.MemAlloc(total + 1)
		if oversized != 0 {
			_ = state.Driver.MemFree(oversized)
		}
		if !driver.IsOutOfMemory(err) || oversized != 0 {
			return fmt.Errorf("physical capacity refusal: pointer=%d error=%v", oversized, err)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	check := func() {
		t.Helper()
		err := worker.Do(t.Context(), func(state *State) error {
			if err := state.Driver.MemsetD32Async(pointer, math.Float32bits(values[0]), uint64(len(values)), state.Stream); err != nil {
				return err
			}
			if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
				return err
			}
			actual := make([]float32, len(values))
			if err := state.Driver.MemcpyDtoH(driver.Bytes(actual), pointer); err != nil {
				return err
			}
			for i := range values {
				if actual[i] != values[i] {
					return fmt.Errorf("CUDA value %d: %g != %g", i, actual[i], values[i])
				}
			}
			stats := state.Driver.MemoryStats()
			bytes := uint64(len(driver.Bytes(values)))
			if stats.CurrentBytes != bytes || stats.PeakBytes != bytes {
				return fmt.Errorf("another process changed local allocation accounting: %+v", stats)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	check()
	if role == "shared-probe" {
		if err := worker.Do(t.Context(), func(state *State) error { return state.Driver.MemFree(pointer) }); err != nil {
			t.Fatal(err)
		}
		if err := worker.Close(); err != nil {
			t.Fatal(err)
		}
		fmt.Println("verified")
		return
	}
	fmt.Println("admitted")
	input := bufio.NewScanner(os.Stdin)
	for input.Scan() {
		switch input.Text() {
		case "check":
			check()
			fmt.Println("checked")
		case "release":
			if err := worker.Do(t.Context(), func(state *State) error { return state.Driver.MemFree(pointer) }); err != nil {
				t.Fatal(err)
			}
			if err := worker.Close(); err != nil {
				t.Fatal(err)
			}
			fmt.Println("released")
		default:
			t.Fatalf("unknown CUDA probe control %q", input.Text())
		}
	}
	if err := input.Err(); err != nil {
		t.Fatal(err)
	}
}
