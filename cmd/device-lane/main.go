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
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

const cudaTestEnv = "OVERGO_CUDA_TEST"

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "device-lane: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	start := time.Now()
	// cuda-info first: it is the availability probe. Failure here means no
	// usable device/driver -- report UNAVAILABLE and exit nonzero so a caller
	// can never mistake absence for green.
	if out, err := command("go", "run", "./cmd/cuda-info"); err != nil {
		fmt.Println("device-lane: UNAVAILABLE -- cuda-info failed; no passing evidence exists")
		fmt.Print(tail(out, 800))
		return err
	}
	steps := [][]string{
		{"go", "run", "./cmd/cuda-smoke"},
		{"go", "test", "./internal/cuda/...", "./internal/model", "./internal/projector", "./internal/optimizer", "-count=1"},
		{"go", "test", "-run", "Device", "./internal/densecausal", "-count=1"},
	}
	for _, step := range steps {
		began := time.Now()
		out, err := commandEnv(append(os.Environ(), cudaTestEnv+"=1"), step[0], step[1:]...)
		fmt.Printf("[device] %-60s %6.1fs %s\n", strings.Join(step[1:], " "), time.Since(began).Seconds(), verdict(err))
		if err != nil {
			fmt.Print(tail(out, 2000))
			return fmt.Errorf("%s failed", strings.Join(step, " "))
		}
	}
	fmt.Printf("=== DEVICE LANE GREEN in %.1fs ===\n", time.Since(start).Seconds())
	fmt.Println("honesty: full device set (manifest-scoped lane is the target form; see docs/MERGE_FLOOR_PLAN.md component 8)")
	return nil
}

func verdict(err error) string {
	if err != nil {
		return "FAIL"
	}
	return "ok"
}

func command(name string, args ...string) (string, error) {
	return commandEnv(os.Environ(), name, args...)
}

func commandEnv(env []string, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func tail(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return "..." + s[len(s)-limit:]
}
