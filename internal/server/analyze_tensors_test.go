package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/inference"
)

// tensorPathGenerator is a fakeGenerator whose model path points at a real GGUF
// fixture, so /analyze/tensors can open and sample it.
type tensorPathGenerator struct {
	*fakeGenerator
	path string
}

func (g tensorPathGenerator) ModelProperties() inference.ModelProperties {
	properties := g.fakeGenerator.ModelProperties()
	properties.Path = g.path
	return properties
}

func writeTensorFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	values := make([]byte, 512*4) // symmetric range [-256, 255]
	for i := 0; i < 512; i++ {
		binary.LittleEndian.PutUint32(values[i*4:], math.Float32bits(float32(i-256)))
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, nil, []gguf.TensorData{
		{Name: "blk.0.weight", Shape: []uint64{512}, Type: gguf.DTypeF32, Data: bytes.NewReader(values)},
	}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestAnalyzeTensorsReturnsDistributionFreeProfiles(t *testing.T) {
	path := writeTensorFixture(t)
	handler := newTestHandler(t, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: path})

	response := serveTestRequest(handler, http.MethodGet, "/analyze/tensors", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result analyzeTensorsResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v body=%s", err, response.Body.String())
	}
	if result.Count != 1 || len(result.Tensors) != 1 {
		t.Fatalf("count = %d / %d, want 1", result.Count, len(result.Tensors))
	}
	if result.Policy.MaxSamplesPerTensor != analyzeTensorMaxSamplesPerTensor {
		t.Errorf("policy sample cap = %d, want %d", result.Policy.MaxSamplesPerTensor, analyzeTensorMaxSamplesPerTensor)
	}
	tensor := result.Tensors[0]
	if tensor.Name != "blk.0.weight" || tensor.Storage != "f32" || tensor.Elements != 512 {
		t.Fatalf("tensor identity = %+v", tensor)
	}
	// Symmetric population: location near -0.5, negligible L-skewness, energy
	// spread across the range.
	if math.Abs(tensor.LMoments.L1-(-0.5)) > 3 {
		t.Errorf("L1 = %.3f, want ~ -0.5", tensor.LMoments.L1)
	}
	if math.Abs(tensor.LMoments.Tau3) > 0.05 {
		t.Errorf("Tau3 = %.4f, want ~ 0", tensor.LMoments.Tau3)
	}
	if tensor.Values.NormalizedEnergyEntropy <= 0 || tensor.Values.NormalizedEnergyEntropy > 1 {
		t.Errorf("energy entropy = %.4f, want in (0,1]", tensor.Values.NormalizedEnergyEntropy)
	}
}

func TestAnalyzeTensorsRejectsNonGet(t *testing.T) {
	handler := newTestHandler(t, tensorPathGenerator{fakeGenerator: &fakeGenerator{}, path: writeTensorFixture(t)})
	response := serveTestRequest(handler, http.MethodPost, "/analyze/tensors", "")
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", response.Code)
	}
}

func TestAnalyzeModelAdvertisesTensorCapability(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/analyze/model", "")
	var result analyzeModelResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !result.Analysis.Tensors {
		t.Error("analysis.tensors = false, want true when model properties are available")
	}
}
