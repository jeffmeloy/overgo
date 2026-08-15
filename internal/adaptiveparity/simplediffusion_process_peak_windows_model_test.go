//go:build windows && modeltest

package adaptiveparity_test

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
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

	candidate := measureModelProcesses(t, simpleDiffusionProcessRuns, root, candidateBinary, "TestRealForwardProcessProbe", "SIMPLEDIFFUSION_PROCESS_PROBE")
	reference := measureModelProcesses(t, simpleDiffusionProcessRuns, filepath.Join(adaptiveGo, "extmodel"), referenceBinary, "TestUViTRealCheckpointVendorParity", "PASS")
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
