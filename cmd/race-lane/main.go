// race-lane: host and device data-race verification, honestly gated.
//
// Host races: the Go race detector (ThreadSanitizer). It requires cgo, so this
// lane sets CGO_ENABLED=1 -- test-only instrumentation, never a runtime
// dependency; the shipped binary stays cgo-free. A missing C compiler is
// UNAVAILABLE and FAILs, never silently skipped.
//
// Device races: CUDA compute-sanitizer (racecheck + synccheck) wrapping a
// compiled device test binary -- the only tool that sees kernel and
// shared-memory hazards, which the Go race detector cannot. A missing GPU/driver
// or a missing compute-sanitizer is UNAVAILABLE and FAILs.
//
// The two tools are complementary: -race cannot see kernels; compute-sanitizer
// cannot see goroutines. One command for the dev agent:
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
	cudaTestEnv   = "OVERGO_CUDA_TEST"
	sanitizerTool = "compute-sanitizer"
	// deviceRacePkg is the package whose device tests are compiled and run under
	// compute-sanitizer. Widen it as more kernel-exercising device tests land.
	deviceRacePkg = "./internal/cuda/executor"
)

// hostRacePkgs are the goroutine-bearing packages worth ThreadSanitizer
// coverage; keep in sync with the host concurrency owners.
var hostRacePkgs = []string{"./internal/server", "./internal/inference", "./internal/model"}

// deviceRaceTools are the compute-sanitizer tools that detect device races:
// racecheck (shared-memory hazards) and synccheck (invalid barrier use).
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

// hostRaceCmd is the ThreadSanitizer command (CGO_ENABLED is supplied via env).
func hostRaceCmd() []string {
	return append([]string{"go", "test", "-race", "-count=1"}, hostRacePkgs...)
}

// sanitizerCmd wraps a compiled binary in compute-sanitizer for one tool.
// --error-exitcode makes a detected hazard a nonzero exit, i.e. a real FAIL.
func sanitizerCmd(tool, bin string, args ...string) []string {
	cmd := []string{sanitizerTool, "--tool", tool, "--error-exitcode", "1", bin}
	return append(cmd, args...)
}

func hostRace() error {
	cc := cCompiler()
	if _, err := exec.LookPath(cc); err != nil {
		return unavailable("host", fmt.Sprintf("C compiler %q not found; the race detector needs cgo", cc))
	}
	cmd := hostRaceCmd()
	began := time.Now()
	out, err := commandEnv(append(os.Environ(), "CGO_ENABLED=1"), cmd[0], cmd[1:]...)
	report("host", strings.Join(hostRacePkgs, " "), began, err)
	if err != nil {
		fmt.Print(tail(out, 2000))
		return fmt.Errorf("host race lane failed")
	}
	return nil
}

func deviceRace() error {
	// cuda-info is the availability probe: failure means no usable device.
	if out, err := command("go", "run", "./cmd/cuda-info"); err != nil {
		fmt.Print(tail(out, 800))
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
	if out, err := command("go", "test", "-c", "-o", bin, deviceRacePkg); err != nil {
		fmt.Print(tail(out, 2000))
		return fmt.Errorf("building device test binary failed")
	}
	for _, tool := range deviceRaceTools {
		cmd := sanitizerCmd(tool, bin, "-test.run", "Device", "-test.count=1")
		began := time.Now()
		out, err := commandEnv(append(os.Environ(), cudaTestEnv+"=1"), cmd[0], cmd[1:]...)
		report("device:"+tool, deviceRacePkg, began, err)
		if err != nil {
			fmt.Print(tail(out, 2000))
			return fmt.Errorf("device %s failed", tool)
		}
	}
	return nil
}

// cCompiler is the C compiler the race detector will invoke: `go env CC`, or gcc.
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

func tail(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return "..." + s[len(s)-limit:]
}
