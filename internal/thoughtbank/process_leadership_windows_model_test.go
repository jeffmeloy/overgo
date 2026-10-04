//go:build windows && modeltest

package thoughtbank

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testprocess"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

const fractaleProcessRuns = 3

func TestFractaleProcessLeadership(t *testing.T) {
	if os.Getenv("OVERGO_FRACTALE_BASELINE") != "1" {
		t.Skip(testskip.Inapplicable + ": set OVERGO_FRACTALE_BASELINE=1 for matched Fractale process evidence")
	}
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	adaptiveGo := testprocess.PinnedTree(t, filepath.Dir(roots.Models), "b8fef3cc227ecb9e942146dc130424029dfe679d", "go")
	temporary := t.TempDir()
	candidateBinary := filepath.Join(temporary, "overgo-fractale.test.exe")
	referenceBinary := filepath.Join(temporary, "adaptive-fractale.test.exe")
	testprocess.BuildTestBinary(t, root, candidateBinary, "modeltest", "./internal/thoughtbank")
	testprocess.BuildTestBinary(t, adaptiveGo, referenceBinary, "research", "./extmodel")

	candidate := testprocess.MeasureTestProcesses(t, fractaleProcessRuns, root, candidateBinary,
		"TestFractaleTwentyTokenProcessProbe", "FRACTALE_PROCESS_PROBE")
	reference := testprocess.MeasureTestProcesses(t, fractaleProcessRuns, filepath.Join(adaptiveGo, "extmodel"), referenceBinary,
		"TestFastWeightBankLMIncrementalDecodeRealCheckpoint", "real-386M")
	candidatePeak, referencePeak := testprocess.MedianProcessPeak(candidate), testprocess.MedianProcessPeak(reference)
	candidateWall, referenceWall := testprocess.MedianProcessWall(candidate), testprocess.MedianProcessWall(reference)
	t.Logf("Fractale candidate=%s reference=%s", testprocess.FormatProcessMeasurements(candidate), testprocess.FormatProcessMeasurements(reference))
	if candidatePeak >= referencePeak {
		t.Fatalf("Fractale process peak %.3f MiB does not beat adaptive %.3f MiB", testprocess.MiB(candidatePeak), testprocess.MiB(referencePeak))
	}
	if candidateWall >= referenceWall {
		t.Fatalf("Fractale process wall %s does not beat adaptive %s", candidateWall, referenceWall)
	}
}
