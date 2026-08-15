//go:build windows && modeltest

package adaptiveparity_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/processmeasure"
	"overgo/internal/testutil"
)

const needleProcessRuns = 3

func TestNeedleProcessPeakLeadership(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	adaptiveRoot := filepath.Dir(roots.Models)
	adaptiveGo := filepath.Join(adaptiveRoot, "go")
	if _, err := os.Stat(filepath.Join(adaptiveGo, "go.mod")); err != nil {
		t.Fatalf("UNAVAILABLE: adaptive Go source absent at %s", adaptiveGo)
	}
	temporary := t.TempDir()
	candidateBinary := filepath.Join(temporary, "overgo-needle.test.exe")
	referenceBinary := filepath.Join(temporary, "adaptive-needle.test.exe")
	buildTestBinary(t, root, candidateBinary, "modeltest", "./internal/seq2seq")
	buildTestBinary(t, adaptiveGo, referenceBinary, "research", "./extmodel")

	candidate := measureNeedleProcesses(t, root, candidateBinary, "TestSingleTokenProcessProbe", "NEEDLE_PROCESS_PROBE")
	reference := measureNeedleProcesses(t, filepath.Join(adaptiveGo, "extmodel"), referenceBinary, "TestNeedleServingGeneratesFromResolvedArtifact", "PASS")
	candidatePeak, referencePeak := minimumPeak(candidate), minimumPeak(reference)
	t.Logf("Needle process peaks candidate=%s reference=%s", formatPeaks(candidate), formatPeaks(reference))
	if candidatePeak >= referencePeak {
		t.Fatalf("Needle process peak %.3f MiB does not beat adaptive %.3f MiB", mib(candidatePeak), mib(referencePeak))
	}
}

func buildTestBinary(t testing.TB, directory, output, tags, pkg string) {
	t.Helper()
	command := exec.Command("go", "test", "-c", "-tags", tags, "-o", output, pkg)
	command.Dir = directory
	if result, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build %s: %v: %s", pkg, err, result)
	}
}

func measureNeedleProcesses(t testing.TB, directory, binary, testName, marker string) []processmeasure.Result {
	t.Helper()
	results := make([]processmeasure.Result, needleProcessRuns)
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

func formatPeaks(results []processmeasure.Result) string {
	values := make([]string, len(results))
	for index, result := range results {
		values[index] = fmt.Sprintf("%.3fMiB/%s", mib(result.PeakWorkingSetByte), result.Wall.Round(time.Millisecond))
	}
	return strings.Join(values, ",")
}

func mib(value uint64) float64 { return float64(value) / (1 << 20) }
