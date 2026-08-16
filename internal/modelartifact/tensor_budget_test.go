package modelartifact

import (
	"bytes"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

// TestMeasurementChargesReadBudget pins the audit's evidence-integrity P1:
// every byte the measurement reads -- sampled or spectral, either format --
// charges the recorded budget through the single read meter. Spectral
// collection can no longer read a full tensor for free, a budget that covers
// sampling but not the spectral read refuses instead of under-reporting, and
// a nonzero spectral policy refuses documents whose measurements carry no
// spectral verdict.
func TestMeasurementChargesReadBudget(t *testing.T) {
	values := make([]float32, 16)
	for i := range values {
		values[i] = float32(i%4) + 1
	}
	file := openGGUF(t, []gguf.TensorData{
		{Name: "matrix", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(values))},
	})
	inventory, err := FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}

	// Sampling alone: 4 samples x 4 bytes = 16 bytes recorded.
	sampled, err := MeasureGGUF(inventory.TensorInventory, file, MeasurementPolicy{
		MaxSamplesPerTensor: 4, MaxReadBytes: 1 << 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Sampling plus spectral: the full 64-byte tensor read must be charged on
	// top of the sampled bytes, so recorded ReadBytes strictly exceeds the
	// sampling-only document's.
	spectral, err := MeasureGGUF(inventory.TensorInventory, file, MeasurementPolicy{
		MaxSamplesPerTensor: 4, MaxReadBytes: 1 << 20, SpectralMaxDim: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	if spectral.Measurements[0].SpectralStatus != SpectralComputed {
		t.Fatalf("spectral status = %q, want computed", spectral.Measurements[0].SpectralStatus)
	}
	if spectral.ReadBytes < sampled.ReadBytes+64 {
		t.Fatalf("spectral read not charged: sampled=%d spectral=%d", sampled.ReadBytes, spectral.ReadBytes)
	}

	// A budget that fits sampling but not the full spectral read refuses.
	if _, err := MeasureGGUF(inventory.TensorInventory, file, MeasurementPolicy{
		MaxSamplesPerTensor: 4, MaxReadBytes: sampled.ReadBytes + 8, SpectralMaxDim: 8,
	}); err == nil || !strings.Contains(err.Error(), "read budget") {
		t.Fatalf("under-budget spectral read accepted: %v", err)
	}

	// A nonzero spectral policy refuses measurements without a verdict.
	bare := spectral.Measurements
	bare[0].SpectralStatus = ""
	bare[0].EffectiveRank = 0
	if _, err := newTensorMeasurementDocument(spectral.Inventory, spectral.Policy, spectral.ReadBytes, bare); err == nil {
		t.Fatal("empty spectral status accepted under a nonzero spectral policy")
	}
}
