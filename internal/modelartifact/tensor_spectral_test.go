package modelartifact

import (
	"bytes"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

func openGGUF(t *testing.T, tensors []gguf.TensorData) *gguf.File {
	t.Helper()
	file, err := gguf.Open(testutil.TempGGUF(t, "fixture.gguf", nil, tensors))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { file.Close() })
	return file
}

func measure(t *testing.T, file *gguf.File, policy MeasurementPolicy) map[string]TensorMeasurement {
	t.Helper()
	inventory, err := FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	document, err := MeasureGGUF(inventory.TensorInventory, file, policy)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]TensorMeasurement{}
	for _, m := range document.Measurements {
		byName[m.Name] = m
	}
	return byName
}

func TestMeasureGGUFComputesEffectiveRank(t *testing.T) {
	ones := make([]float32, 16) // 4x4 all-ones -> rank 1 -> effective rank 1/4
	for i := range ones {
		ones[i] = 1
	}
	identity := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} // 4x4 -> full rank -> 1
	bias := []float32{1, 2, 3, 4}                                         // 1-D -> not applicable
	file := openGGUF(t, []gguf.TensorData{
		{Name: "rank1", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(ones))},
		{Name: "full", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(identity))},
		{Name: "bias", Shape: []uint64{4}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(bias))},
	})
	got := measure(t, file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20, SpectralMaxDim: 8})

	if m := got["rank1"]; m.SpectralStatus != SpectralComputed || math.Abs(m.EffectiveRank-0.25) > 1e-6 {
		t.Errorf("rank1: status=%q effective_rank=%.6f, want computed 0.25", m.SpectralStatus, m.EffectiveRank)
	}
	if m := got["full"]; m.SpectralStatus != SpectralComputed || math.Abs(m.EffectiveRank-1) > 1e-6 {
		t.Errorf("full: status=%q effective_rank=%.6f, want computed 1.0", m.SpectralStatus, m.EffectiveRank)
	}
	if m := got["bias"]; m.SpectralStatus != SpectralNotApplicable || m.EffectiveRank != 0 {
		t.Errorf("bias (1-D): status=%q effective_rank=%.6f, want not-applicable 0", m.SpectralStatus, m.EffectiveRank)
	}
}

func TestMeasureGGUFEffectiveRankIsOrderSensitive(t *testing.T) {
	// [[2,0],[0,1]] has singular values {2,1} -> effective rank ~0.9449, a value a
	// scrambled element order would not reproduce. Guards the sequential full read
	// (fullGGUFTensorValues == sampleGGUFTensorValues at full coverage).
	file := openGGUF(t, []gguf.TensorData{
		{Name: "diag", Shape: []uint64{2, 2}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE([]float32{2, 0, 0, 1}))},
	})
	got := measure(t, file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20, SpectralMaxDim: 8})
	if m := got["diag"]; m.SpectralStatus != SpectralComputed || math.Abs(m.EffectiveRank-0.9449) > 1e-3 {
		t.Errorf("diag: status=%q effective_rank=%.5f, want computed ~0.9449", m.SpectralStatus, m.EffectiveRank)
	}
}

func TestMeasureGGUFDefersOversizeSpectral(t *testing.T) {
	big := make([]float32, 64) // 8x8, budget 4 -> deferred
	for i := range big {
		big[i] = float32(i % 7)
	}
	file := openGGUF(t, []gguf.TensorData{
		{Name: "big", Shape: []uint64{8, 8}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(big))},
	})
	got := measure(t, file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20, SpectralMaxDim: 4})
	if m := got["big"]; m.SpectralStatus != SpectralDeferred || m.EffectiveRank != 0 {
		t.Errorf("big: status=%q effective_rank=%.6f, want deferred 0", m.SpectralStatus, m.EffectiveRank)
	}
}

func TestMeasureGGUFSpectralOffByDefault(t *testing.T) {
	weight := make([]float32, 16)
	file := openGGUF(t, []gguf.TensorData{
		{Name: "w", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(testutil.Float32LE(weight))},
	})
	got := measure(t, file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20})
	if m := got["w"]; m.SpectralStatus != "" || m.EffectiveRank != 0 {
		t.Errorf("spectral must be off by default: status=%q effective_rank=%.6f", m.SpectralStatus, m.EffectiveRank)
	}
}
