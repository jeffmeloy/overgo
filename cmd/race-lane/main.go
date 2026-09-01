// race-lane: host ThreadSanitizer plus CUDA racecheck/synccheck.
// Missing compiler, device, driver, or sanitizer fails as UNAVAILABLE.
//
//	go run ./cmd/race-lane            # both lanes
//	go run ./cmd/race-lane -host      # host goroutine races only
//	go run ./cmd/race-lane -device    # device/kernel races only
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/runrecord"
)

const (
	cudaTestEnv = "OVERGO_CUDA_TEST"
	// raceCompilerEnv names an explicit cgo compiler for the host lane, the
	// same environment-gate pattern the browser lane uses: the worktree-local
	// toolchain is machine state a gate's immutable candidate worktree does
	// not carry, so a verifier running there passes the real location in.
	raceCompilerEnv   = "OVERGO_RACE_CC"
	hostRacePattern   = "./internal/..."
	sanitizerTool     = "compute-sanitizer"
	deviceRacePackage = "./internal/cuda/executor"
	localRaceCompiler = ".tools/llvm-mingw/bin/x86_64-w64-mingw32-clang.exe"
	// Race instrumentation makes the real synthetic VAE workload exceed Go's
	// default package timeout on the reference Windows host. Keep the full
	// workload and give every package a bounded, evidence-derived allowance:
	// the 2026-08-31 campaign run cut internal/tabularicl at exactly 30
	// minutes with its late decoder-trainer tests still queued, so the
	// measured need is above 30 and the allowance doubles it.
	hostRaceTimeout = 60 * time.Minute
)

var deviceRaceTools = []string{"racecheck", "synccheck"}

func main() {
	host := flag.Bool("host", false, "run only the host goroutine race lane")
	device := flag.Bool("device", false, "run only the device/kernel race lane")
	flag.Parse()
	if err := run(*host, *device); err != nil {
		fmt.Fprintf(os.Stderr, "race-lane: %v\n", err)
		os.Exit(1)
	}
}

func run(hostOnly, deviceOnly bool) error {
	runHost, runDevice := hostOnly, deviceOnly
	if !hostOnly && !deviceOnly {
		runHost, runDevice = true, true
	}
	start := time.Now()
	if runHost {
		if err := hostRace(); err != nil {
			return err
		}
	}
	if runDevice {
		if err := deviceRace(); err != nil {
			return err
		}
	}
	fmt.Printf("=== RACE LANE GREEN in %.1fs ===\n", time.Since(start).Seconds())
	fmt.Printf("honesty: host-race %s, device-sanitizer %s\n", ranLabel(runHost), ranLabel(runDevice))
	return nil
}

func sanitizerCmd(tool, bin string, args ...string) []string {
	cmd := []string{sanitizerTool, "--tool", tool, "--error-exitcode", "1", bin}
	return append(cmd, args...)
}

func hostRace() error {
	cc, err := cCompiler()
	if err != nil {
		return unavailable("host", err.Error())
	}
	cmd := []string{"go", "test", "-race", "-count=1", "-timeout", hostRaceTimeout.String(), hostRacePattern}
	began := time.Now()
	out, err := clioptions.CombinedOutput(append(os.Environ(), "CGO_ENABLED=1", "CC="+cc), cmd[0], cmd[1:]...)
	report("host", hostRacePattern, began, err)
	if err != nil {
		fmt.Print(out)
		return runrecord.LaneError(runrecord.LaneFailed, "host race lane failed")
	}
	return nil
}

func deviceRace() error {
	// cuda-info is the availability probe: failure means no usable device.
	if out, err := clioptions.CombinedOutput(os.Environ(), "go", "run", "./cmd/cuda-info"); err != nil {
		fmt.Print(out)
		return unavailable("device", "cuda-info failed; no usable GPU/driver")
	}
	if _, err := exec.LookPath(sanitizerTool); err != nil {
		return unavailable("device", sanitizerTool+" not on PATH; install the CUDA toolkit sanitizer")
	}
	dir, err := os.MkdirTemp("", "racelane")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	bin := filepath.Join(dir, "cudarace.test")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := clioptions.CombinedOutput(os.Environ(), "go", "test", "-c", "-o", bin, deviceRacePackage); err != nil {
		fmt.Print(out)
		return runrecord.LaneError(runrecord.LaneFailed, "building device test binary failed")
	}
	for _, tool := range deviceRaceTools {
		cmd := sanitizerCmd(tool, bin, "-test.run", "Device", "-test.count=1")
		began := time.Now()
		out, err := clioptions.CombinedOutput(append(os.Environ(), cudaTestEnv+"=1"), cmd[0], cmd[1:]...)
		report("device:"+tool, deviceRacePackage, began, err)
		if err != nil {
			fmt.Print(out)
			return runrecord.LaneError(runrecord.LaneFailed, "device "+tool+" failed")
		}
	}
	return nil
}

func cCompiler() (string, error) {
	if override := strings.TrimSpace(os.Getenv(raceCompilerEnv)); override != "" {
		if resolved, findErr := exec.LookPath(override); findErr == nil {
			return resolved, nil
		}
		return "", fmt.Errorf("c compiler unavailable (%s names %q, which is absent); the race detector needs cgo", raceCompilerEnv, override)
	}
	out, err := clioptions.CombinedOutput(os.Environ(), "go", "env", "CC")
	configured := "gcc"
	if err == nil {
		if cc := strings.TrimSpace(out); cc != "" {
			configured = cc
		}
	}
	moduleRoot := ""
	if out, err = clioptions.CombinedOutput(os.Environ(), "go", "env", "GOMOD"); err == nil {
		if module := strings.TrimSpace(out); module != "" && module != os.DevNull {
			moduleRoot = filepath.Dir(module)
		}
	}
	return selectCompiler(configured, moduleRoot, exec.LookPath)
}

func selectCompiler(configured, moduleRoot string, lookPath func(string) (string, error)) (string, error) {
	candidates := []string{configured}
	if moduleRoot != "" {
		candidates = append(candidates, filepath.Join(moduleRoot, filepath.FromSlash(localRaceCompiler)))
	}
	for _, candidate := range candidates {
		if resolved, findErr := lookPath(candidate); findErr == nil {
			return resolved, nil
		}
	}
	return "", fmt.Errorf("c compiler unavailable (checked configured %q and worktree-local %q); the race detector needs cgo", configured, localRaceCompiler)
}

func unavailable(lane, why string) error {
	fmt.Printf("race-lane: %s UNAVAILABLE -- %s; no passing evidence exists\n", lane, why)
	return runrecord.LaneError(runrecord.LaneUnavailable, lane+" lane unavailable")
}

func ranLabel(ran bool) string {
	if ran {
		return "ran"
	}
	return "SKIPPED (flag)"
}

func report(name, detail string, began time.Time, err error) {
	fmt.Printf("[%-14s] %-45s %6.1fs %s\n", name, detail, time.Since(began).Seconds(), clioptions.Verdict(err))
}
