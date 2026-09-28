//go:build windows

package inference

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
)

// TestDeviceMemoryOwners: after a device generation the owners account for
// every live byte the runner holds, weights among them, within the device's
// driver-reported total.
func TestDeviceMemoryOwners(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openF32FixtureRunner(testutil.HermeticLlamaGGUF(t, hermeticContext), OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	ctx := t.Context()
	if _, _, err := runner.Generate(ctx, "ab", GenerateOptions{MaxNewTokens: 4}); err != nil {
		t.Fatal(err)
	}
	memory, err := runner.DeviceMemory(ctx)
	if err != nil {
		t.Fatal(err)
	}
	stats, err := runner.DeviceMemoryStats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var owned, weights uint64
	for _, owner := range memory.Owners {
		owned += owner.Bytes
		if owner.Name == "weights" {
			weights = owner.Bytes
		}
	}
	if owned != stats.CurrentBytes || weights == 0 || memory.TotalBytes == 0 || memory.FreeBytes > memory.TotalBytes || owned > memory.TotalBytes {
		t.Fatalf("owners %+v sum %d (weights %d) of live %d; device free %d of %d", memory.Owners, owned, weights, stats.CurrentBytes, memory.FreeBytes, memory.TotalBytes)
	}
	t.Logf("owners %+v; device free %d of %d", memory.Owners, memory.FreeBytes, memory.TotalBytes)
}
