//go:build windows && modeltest

package adaptiveparity_test

import (
	"context"
	"fmt"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/processmeasure"
)

func buildTestBinary(t testing.TB, directory, output, tags, pkg string) {
	t.Helper()
	command := exec.Command("go", "test", "-c", "-tags", tags, "-o", output, pkg)
	command.Dir = directory
	if result, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v: %s", pkg, err, result)
	}
}

func measureModelProcesses(t testing.TB, runs int, directory, binary, testName, marker string) []processmeasure.Result {
	t.Helper()
	results := make([]processmeasure.Result, runs)
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

func minimumPeak(results []processmeasure.Result) uint64 {
	best := results[0].PeakWorkingSetByte
	for _, result := range results[1:] {
		best = min(best, result.PeakWorkingSetByte)
	}
	return best
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

func formatPeaks(results []processmeasure.Result) string {
	values := make([]string, len(results))
	for index, result := range results {
		values[index] = fmt.Sprintf("%.3fMiB/%s", mib(result.PeakWorkingSetByte), result.Wall.Round(time.Millisecond))
	}
	return strings.Join(values, ",")
}

func mib(value uint64) float64 { return float64(value) / (1 << 20) }
