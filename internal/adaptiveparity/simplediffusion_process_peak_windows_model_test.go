//go:build windows && modeltest

package adaptiveparity_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/processmeasure"
	"overgo/internal/testutil"
)

const simpleDiffusionProcessRuns = 3

func TestSimpleDiffusionProcessLeadership(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	adaptiveGo := filepath.Join(filepath.Dir(roots.Models), "go")
	if _, err := os.Stat(filepath.Join(adaptiveGo, "go.mod")); err != nil {
		t.Fatalf("UNAVAILABLE: adaptive Go source absent at %s", adaptiveGo)
	}
	temporary := t.TempDir()
	candidateBinary := filepath.Join(temporary, "overgo-simplediffusion.test.exe")
	referenceBinary := filepath.Join(temporary, "adaptive-simplediffusion.test.exe")
	buildTestBinary(t, root, candidateBinary, "modeltest", "./internal/diffusionimage")
	buildTestBinary(t, adaptiveGo, referenceBinary, "", "./extmodel")

	candidate := measureSimpleDiffusionProcesses(t, root, candidateBinary, "TestRealForwardProcessProbe", "SIMPLEDIFFUSION_PROCESS_PROBE")
	reference := measureSimpleDiffusionProcesses(t, filepath.Join(adaptiveGo, "extmodel"), referenceBinary, "TestUViTRealCheckpointVendorParity", "PASS")
	candidatePeak, referencePeak := medianProcessPeak(candidate), medianProcessPeak(reference)
	candidateWall, referenceWall := medianProcessWall(candidate), medianProcessWall(reference)
	t.Logf("SimpleDiffusion candidate=%s reference=%s", formatPeaks(candidate), formatPeaks(reference))
	if candidatePeak >= referencePeak {
		t.Fatalf("SimpleDiffusion process peak %.3f MiB does not beat adaptive %.3f MiB", mib(candidatePeak), mib(referencePeak))
	}
	if candidateWall >= referenceWall {
		t.Fatalf("SimpleDiffusion process wall %s does not beat adaptive %s", candidateWall, referenceWall)
	}
}

func measureSimpleDiffusionProcesses(t testing.TB, directory, binary, testName, marker string) []processmeasure.Result {
	t.Helper()
	results := make([]processmeasure.Result, simpleDiffusionProcessRuns)
	for index := range results {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		command := exec.CommandContext(ctx, binary, "-test.run=^"+testName+"$", "-test.v")
		command.Dir = directory
		result, err := processmeasure.Measure(command)
		cancel()
		if err != nil {
			t.Fatalf("%s run %d: %v: %s", testName, index, err, result.Output)
		}
		if !strings.Contains(string(result.Output), marker) {
			t.Fatalf("%s run %d lacks %q: %s", testName, index, marker, result.Output)
		}
		results[index] = result
	}
	return results
}

func medianProcessWall(results []processmeasure.Result) time.Duration {
	walls := make([]time.Duration, len(results))
	for index, result := range results {
		walls[index] = result.Wall
	}
	slices.Sort(walls)
	return walls[len(walls)/2]
}

func medianProcessPeak(results []processmeasure.Result) uint64 {
	peaks := make([]uint64, len(results))
	for index, result := range results {
		peaks[index] = result.PeakWorkingSetByte
	}
	slices.Sort(peaks)
	return peaks[len(peaks)/2]
}
