package model_test

import (
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// realCatalog reads a registered GGUF's spec and weight catalog the way
// serving reads them, without loading tensor data.
func realCatalog(t *testing.T, path string) (model.Spec, model.Weights) {
	t.Helper()
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	spec := testcheck.Must(model.ReadSpec(file))(t)
	weights := testcheck.Must(model.ReadWeights(file, spec))(t)
	return spec, weights
}

// TestReadRealQwen35Catalog holds the validated Qwen3.5 fixture's 32-block
// hybrid catalog: recurrent blocks with a full-attention block every fourth.
func TestReadRealQwen35Catalog(t *testing.T) {
	testskip.Short(t)
	spec, weights := realCatalog(t, modeltest.Directory(t, "Qwen3.5-9B-Q8_0.gguf"))
	if spec.Architecture != "qwen35" || len(weights.Layers) != 32 ||
		!weights.Layers[0].Recurrent || weights.Layers[3].Recurrent {
		t.Fatalf("unexpected real Qwen3.5 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}
