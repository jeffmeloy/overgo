//go:build windows && modeltest

package thoughtbank

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testutil"
)

const fractaleProcessRuns = 3

func TestFractaleProcessLeadership(t *testing.T) {
	if os.Getenv("OVERGO_FRACTALE_BASELINE") != "1" {
		t.Skip("set OVERGO_FRACTALE_BASELINE=1 for matched Fractale process evidence")
	}
	root := testutil.RepoRoot(t)
	adaptiveGo := os.Getenv("OVERGO_FRACTALE_ADAPTIVE_GO")
	if adaptiveGo == "" {
		adaptiveGo = filepath.Join(root, "build", "adaptive-b8fef3cc2", "go")
	}
	if _, err := os.Stat(filepath.Join(adaptiveGo, "go.mod")); err != nil {
		t.Fatalf("UNAVAILABLE: retained adaptive Fractale source absent at %s", adaptiveGo)
	}
	temporary := t.TempDir()
	candidateBinary := filepath.Join(temporary, "overgo-fractale.test.exe")
	referenceBinary := filepath.Join(temporary, "adaptive-fractale.test.exe")
	testutil.BuildTestBinary(t, root, candidateBinary, "modeltest", "./internal/thoughtbank")
	testutil.BuildTestBinary(t, adaptiveGo, referenceBinary, "research", "./extmodel")

	candidate := testutil.MeasureTestProcesses(t, fractaleProcessRuns, root, candidateBinary,
		"TestFractaleTwentyTokenProcessProbe", "FRACTALE_PROCESS_PROBE")
	reference := testutil.MeasureTestProcesses(t, fractaleProcessRuns, filepath.Join(adaptiveGo, "extmodel"), referenceBinary,
		"TestFastWeightBankLMIncrementalDecodeRealCheckpoint", "real-386M")
	candidatePeak, referencePeak := testutil.MedianProcessPeak(candidate), testutil.MedianProcessPeak(reference)
	candidateWall, referenceWall := testutil.MedianProcessWall(candidate), testutil.MedianProcessWall(reference)
	t.Logf("Fractale candidate=%s reference=%s", testutil.FormatProcessMeasurements(candidate), testutil.FormatProcessMeasurements(reference))
	if candidatePeak >= referencePeak {
		t.Fatalf("Fractale process peak %.3f MiB does not beat adaptive %.3f MiB", testutil.MiB(candidatePeak), testutil.MiB(referencePeak))
	}
	if candidateWall >= referenceWall {
		t.Fatalf("Fractale process wall %s does not beat adaptive %s", candidateWall, referenceWall)
	}
}
