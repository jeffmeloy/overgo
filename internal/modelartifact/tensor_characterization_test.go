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

func f32GGUF(t *testing.T, tensors []gguf.TensorData) *gguf.File {
	t.Helper()
	path := filepath.Join(t.TempDir(), "characterize.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, nil, tensors, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	opened, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { opened.Close() })
	return opened
}

func f32Bytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func TestCharacterizeGGUFTensors(t *testing.T) {
	weight := make([]float32, 512) // symmetric range [-256, 255]
	for i := range weight {
		weight[i] = float32(i - 256)
	}
	bias := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	file := f32GGUF(t, []gguf.TensorData{
		{Name: "weight", Shape: []uint64{512}, Type: gguf.DTypeF32, Data: bytes.NewReader(f32Bytes(weight))},
		{Name: "bias", Shape: []uint64{8}, Type: gguf.DTypeF32, Data: bytes.NewReader(f32Bytes(bias))},
	})

	profiles, err := CharacterizeGGUFTensors(file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	if len(profiles) != 2 {
		t.Fatalf("got %d profiles, want 2", len(profiles))
	}

	byName := map[string]TensorCharacterization{}
	for _, p := range profiles {
		byName[p.Name] = p
	}

	w := byName["weight"]
	if w.Storage != "f32" || len(w.Shape) != 1 || w.Shape[0] != 512 || w.Elements != 512 {
		t.Fatalf("weight identity wrong: %+v", w)
	}
	if w.FiniteSamples == 0 || w.FiniteSamples != w.Samples {
		t.Errorf("weight finite samples = %d of %d, want all finite", w.FiniteSamples, w.Samples)
	}
	// Symmetric population: location near -0.5, negligible L-skewness.
	if math.Abs(w.LMoments.L1-(-0.5)) > 3 {
		t.Errorf("weight L1 = %.3f, want ~ -0.5", w.LMoments.L1)
	}
	if math.Abs(w.LMoments.Tau3) > 0.05 {
		t.Errorf("weight Tau3 = %.4f, want ~ 0 (symmetric)", w.LMoments.Tau3)
	}
	if w.Values.MaxAbsolute < 250 {
		t.Errorf("weight max-abs = %.1f, want ~256", w.Values.MaxAbsolute)
	}

	b := byName["bias"]
	if b.Elements != 8 || b.Samples != 8 {
		t.Fatalf("bias counts = elements %d samples %d, want 8/8", b.Elements, b.Samples)
	}
	if math.Abs(b.Median-4.5) > 1e-6 || math.Abs(b.LMoments.L1-4.5) > 1e-6 {
		t.Errorf("bias median=%.4f L1=%.4f, want 4.5/4.5", b.Median, b.LMoments.L1)
	}
}

func TestCharacterizeGGUFTensorsReadBudget(t *testing.T) {
	weight := make([]float32, 512)
	file := f32GGUF(t, []gguf.TensorData{
		{Name: "weight", Shape: []uint64{512}, Type: gguf.DTypeF32, Data: bytes.NewReader(f32Bytes(weight))},
	})
	// A read budget below one sampled tensor must be refused, not truncated.
	_, err := CharacterizeGGUFTensors(file, MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 4})
	if err == nil {
		t.Fatal("expected read-budget error")
	}
}
