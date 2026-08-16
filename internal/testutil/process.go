package testutil

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

// BuildTestBinary compiles one isolated test executable.
func BuildTestBinary(t testing.TB, directory, output, tags, pkg string) {
	t.Helper()
	command := exec.Command("go", "test", "-c", "-tags", tags, "-o", output, pkg)
	command.Dir = directory
	if result, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v: %s", pkg, err, result)
	}
}

// MeasureTestProcesses runs identical isolated probes.
func MeasureTestProcesses(t testing.TB, runs int, directory, binary, testName, marker string) []processmeasure.Result {
	t.Helper()
	results := make([]processmeasure.Result, runs)
	for index := range results {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
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

func MedianProcessWall(results []processmeasure.Result) time.Duration {
	values := make([]time.Duration, len(results))
	for index, result := range results {
		values[index] = result.Wall
	}
	slices.Sort(values)
	return values[len(values)/2]
}

func MedianProcessPeak(results []processmeasure.Result) uint64 {
	values := make([]uint64, len(results))
	for index, result := range results {
		values[index] = result.PeakWorkingSetByte
	}
	slices.Sort(values)
	return values[len(values)/2]
}

func FormatProcessMeasurements(results []processmeasure.Result) string {
	values := make([]string, len(results))
	for index, result := range results {
		values[index] = fmt.Sprintf("%.3fMiB/%s", MiB(result.PeakWorkingSetByte), result.Wall.Round(time.Millisecond))
	}
	return strings.Join(values, ",")
}

func MiB(value uint64) float64 { return float64(value) / (1 << 20) }
