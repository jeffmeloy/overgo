//go:build windows && modeltest

package adaptiveparity_test

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
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

	candidate := measureModelProcesses(t, needleProcessRuns, root, candidateBinary, "TestSingleTokenProcessProbe", "NEEDLE_PROCESS_PROBE")
	reference := measureModelProcesses(t, needleProcessRuns, filepath.Join(adaptiveGo, "extmodel"), referenceBinary, "TestNeedleServingGeneratesFromResolvedArtifact", "PASS")
	candidatePeak, referencePeak := minimumPeak(candidate), minimumPeak(reference)
	t.Logf("Needle process peaks candidate=%s reference=%s", formatPeaks(candidate), formatPeaks(reference))
	if candidatePeak >= referencePeak {
		t.Fatalf("Needle process peak %.3f MiB does not beat adaptive %.3f MiB", mib(candidatePeak), mib(referencePeak))
	}
}
