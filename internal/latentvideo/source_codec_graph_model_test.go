//go:build modeltest

package latentvideo

import (
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/pytorchzip"
	"overgo/internal/testutil"
)

func TestLiveEditSourceCodecGraph(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := filepath.Join(roots.Models, "Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		t.Fatalf("UNAVAILABLE: Wan VAE metadata: %v", err)
	}
	metas := catalog.Tensors
	plan, err := CompileVAEEncoderPlan(metas)
	if err != nil {
		t.Fatal(err)
	}
	prefixes := plan.OpPrefixes()
	if plan.InputChannels != 3 || plan.LatentChannels != 16 || plan.MomentChannels != 32 ||
		plan.Stride != [3]int{4, 8, 8} || plan.Ops() != 17 ||
		len(prefixes) != 17 || prefixes[0] != "encoder.conv1" || prefixes[15] != "encoder.head" || prefixes[16] != "conv1" ||
		plan.UsedTensorCount == 0 || plan.UsedWeightBytes <= 0 || plan.LargestOpWeightBytes <= 0 {
		t.Fatalf("encoder plan = %+v prefixes=%v", plan, prefixes)
	}
	t.Logf("real Wan source encoder: ops=%d tensors=%d weights=%.3fMiB max_op=%.3fMiB stride=%v",
		plan.Ops(), plan.UsedTensorCount, float64(plan.UsedWeightBytes)/(1<<20), float64(plan.LargestOpWeightBytes)/(1<<20), plan.Stride)
}
