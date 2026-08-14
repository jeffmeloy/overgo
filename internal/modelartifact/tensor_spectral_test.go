package modelartifact

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
)

func spectralF32(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func openSpectralGGUF(t *testing.T, tensors []gguf.TensorData) *gguf.File {
	t.Helper()
	path := filepath.Join(t.TempDir(), "spectral.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(f, nil, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	opened, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { opened.Close() })
	return opened
}

func measure(t *testing.T, file *gguf.File, policy MeasurementPolicy) map[string]TensorMeasurement {
	t.Helper()
	inventory, err := FromGGUF(file)
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
	file := openSpectralGGUF(t, []gguf.TensorData{
		{Name: "rank1", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(spectralF32(ones))},
		{Name: "full", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(spectralF32(identity))},
		{Name: "bias", Shape: []uint64{4}, Type: gguf.DTypeF32, Data: bytes.NewReader(spectralF32(bias))},
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

func TestMeasureGGUFDefersOversizeSpectral(t *testing.T) {
	big := make([]float32, 64) // 8x8, budget 4 -> deferred
	for i := range big {
		big[i] = float32(i % 7)
	}
	file := openSpectralGGUF(t, []gguf.TensorData{
		{Name: "big", Shape: []uint64{8, 8}, Type: gguf.DTypeF32, Data: bytes.NewReader(spectralF32(big))},
	})
	got := measure(t, file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20, SpectralMaxDim: 4})
	if m := got["big"]; m.SpectralStatus != SpectralDeferred || m.EffectiveRank != 0 {
		t.Errorf("big: status=%q effective_rank=%.6f, want deferred 0", m.SpectralStatus, m.EffectiveRank)
	}
}

func TestMeasureGGUFSpectralOffByDefault(t *testing.T) {
	weight := make([]float32, 16)
	file := openSpectralGGUF(t, []gguf.TensorData{
		{Name: "w", Shape: []uint64{4, 4}, Type: gguf.DTypeF32, Data: bytes.NewReader(spectralF32(weight))},
	})
	got := measure(t, file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20})
	if m := got["w"]; m.SpectralStatus != "" || m.EffectiveRank != 0 {
		t.Errorf("spectral must be off by default: status=%q effective_rank=%.6f", m.SpectralStatus, m.EffectiveRank)
	}
}
