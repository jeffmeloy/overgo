// Package testprocess builds and measures isolated test processes; it
// lives apart from testutil so the packages that only compare numbers
// inherit no program-running reach from their test helpers.
package testprocess

import (
	"fmt"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/processmeasure"
)

// BuildTestBinary compiles one isolated test executable.
func BuildTestBinary(t testing.TB, directory, output, tags, pkg string) {
	t.Helper()
	command := exec.Command("go", "test", "-c", "-tags", tags, "-o", output, pkg)
	command.Dir = directory
	if result, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v: %s", pkg, err, result)
	}
}

// MeasureTestProcesses runs identical isolated probes; each probe runs
// under the calling test's context, so the test's own end bounds it.
func MeasureTestProcesses(t testing.TB, runs int, directory, binary, testName, marker string) []processmeasure.Result {
	t.Helper()
	results := make([]processmeasure.Result, runs)
	for index := range results {
		command := exec.CommandContext(t.Context(), binary, "-test.run=^"+testName+"$", "-test.v")
		command.Dir = directory
		result, err := processmeasure.Measure(command)
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

// MedianProcessWall is the median wall time of the measured runs.
func MedianProcessWall(results []processmeasure.Result) time.Duration {
	values := make([]time.Duration, len(results))
	for index, result := range results {
		values[index] = result.Wall
	}
	slices.Sort(values)
	return values[len(values)/2]
}

// MedianProcessPeak is the median peak working set of the measured runs.
func MedianProcessPeak(results []processmeasure.Result) uint64 {
	values := make([]uint64, len(results))
	for index, result := range results {
		values[index] = result.PeakWorkingSetByte
	}
	slices.Sort(values)
	return values[len(values)/2]
}

// FormatProcessMeasurements renders each run as peak and wall.
func FormatProcessMeasurements(results []processmeasure.Result) string {
	values := make([]string, len(results))
	for index, result := range results {
		values[index] = fmt.Sprintf("%.3fMiB/%s", MiB(result.PeakWorkingSetByte), result.Wall.Round(time.Millisecond))
	}
	return strings.Join(values, ",")
}

// MiB converts bytes to mebibytes.
func MiB(value uint64) float64 { return float64(value) / (1 << 20) }
