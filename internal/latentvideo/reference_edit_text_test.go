package latentvideo

import (
	"math"
	"path/filepath"
	"testing"
)

func TestLiveEditTextConditioning(t *testing.T) {
	modelDirectory := wanModelDir(t)
	base, err := LoadDenoiserConfig(modelDirectory, referenceDenoiserPolicy)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := CompileReferenceEditCheckpoint(filepath.Join(filepath.Dir(modelDirectory), "LiveEdit", "ar-forcing_002000.pt"), base)
	if err != nil {
		t.Fatal(err)
	}
	weights, sourceBytes, err := checkpoint.loadTextProjection()
	if err != nil {
		t.Fatal(err)
	}
	const textDim = 4096
	context := make([]float32, textDim)
	for index := range context {
		context[index] = float32(math.Sin(float64(index) * 0.01))
	}
	projected, err := projectCompactTextConditioning(context, 1, checkpoint.Config.TextLen, textDim, checkpoint.Config.Dim, weights, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(projected) != checkpoint.Config.TextLen*checkpoint.Config.Dim || sourceBytes <= 0 {
		t.Fatalf("projected=%d source_bytes=%d", len(projected), sourceBytes)
	}
	for index, value := range projected {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("projected[%d]=%g", index, value)
		}
	}
	if projected[checkpoint.Config.Dim] != projected[2*checkpoint.Config.Dim] {
		t.Fatal("padding rows do not share the projected inactive row")
	}
	t.Logf("LiveEdit text projection rows=%d dim=%d source_bytes=%d", checkpoint.Config.TextLen, checkpoint.Config.Dim, sourceBytes)
}
