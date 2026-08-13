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
)

const (
	cudaTestEnv       = "OVERGO_CUDA_TEST"
	hostRacePattern   = "./internal/..."
	sanitizerTool     = "compute-sanitizer"
	deviceRacePackage = "./internal/cuda/executor"
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
	cc := cCompiler()
	if _, err := exec.LookPath(cc); err != nil {
		return unavailable("host", fmt.Sprintf("C compiler %q not found; the race detector needs cgo", cc))
	}
	cmd := []string{"go", "test", "-race", "-count=1", hostRacePattern}
	began := time.Now()
	out, err := commandEnv(append(os.Environ(), "CGO_ENABLED=1"), cmd[0], cmd[1:]...)
	report("host", hostRacePattern, began, err)
	if err != nil {
		fmt.Print(out)
		return fmt.Errorf("host race lane failed")
	}
	return nil
}

func deviceRace() error {
	// cuda-info is the availability probe: failure means no usable device.
	if out, err := command("go", "run", "./cmd/cuda-info"); err != nil {
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
	if out, err := command("go", "test", "-c", "-o", bin, deviceRacePackage); err != nil {
		fmt.Print(out)
		return fmt.Errorf("building device test binary failed")
	}
	for _, tool := range deviceRaceTools {
		cmd := sanitizerCmd(tool, bin, "-test.run", "Device", "-test.count=1")
		began := time.Now()
		out, err := commandEnv(append(os.Environ(), cudaTestEnv+"=1"), cmd[0], cmd[1:]...)
		report("device:"+tool, deviceRacePackage, began, err)
		if err != nil {
			fmt.Print(out)
			return fmt.Errorf("device %s failed", tool)
		}
	}
	return nil
}

func cCompiler() string {
	out, err := command("go", "env", "CC")
	if err == nil {
		if cc := strings.TrimSpace(out); cc != "" {
			return cc
		}
	}
	return "gcc"
}

func unavailable(lane, why string) error {
	fmt.Printf("race-lane: %s UNAVAILABLE -- %s; no passing evidence exists\n", lane, why)
	return fmt.Errorf("%s lane unavailable", lane)
}

func ranLabel(ran bool) string {
	if ran {
		return "ran"
	}
	return "SKIPPED (flag)"
}

func report(name, detail string, began time.Time, err error) {
	fmt.Printf("[%-14s] %-45s %6.1fs %s\n", name, detail, time.Since(began).Seconds(), verdict(err))
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
