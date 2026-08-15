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
	"strings"
	"time"

	"overgo/internal/clioptions"
)

const cudaTestEnv = "OVERGO_CUDA_TEST"

func main() {
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
		return err
	}
	steps := [][]string{
		{"go", "run", "./cmd/cuda-smoke"},
		{"go", "test", "./internal/cuda/...", "./internal/model", "./internal/projector", "./internal/optimizer", "./internal/devicemath", "-count=1"},
		{"go", "test", "-run", "Device", "./internal/densecausal", "-count=1"},
	}
	for _, step := range steps {
		began := time.Now()
		out, err := clioptions.CombinedOutput(append(os.Environ(), cudaTestEnv+"=1"), step[0], step[1:]...)
		fmt.Printf("[device] %-60s %6.1fs %s\n", strings.Join(step[1:], " "), time.Since(began).Seconds(), verdict(err))
		if err != nil {
			fmt.Print(clioptions.Tail(out, 2000))
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
