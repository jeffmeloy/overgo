//go:build integration

package latentvideo

import (
	"math"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
)

func TestLiveEditTextConditioning(t *testing.T) {
	modelDirectory := wanModelDir(t)
	base := testcheck.Must(LoadDenoiserConfig(modelDirectory, referenceDenoiserPolicy))(t)
	checkpoint := testcheck.Must(CompileReferenceEditCheckpoint(filepath.Join(filepath.Dir(modelDirectory), "LiveEdit", "ar-forcing_002000.pt"), base))(t)
	weights, sourceBytes, err := checkpoint.loadTextProjection()
	testcheck.NoError(t, err)
	const textDim = 4096
	context := make([]float32, textDim)
	for index := range context {
		context[index] = float32(math.Sin(float64(index) * 0.01))
	}
	projected := testcheck.Must(projectCompactTextConditioning(context, 1, checkpoint.Config.TextLen, textDim, checkpoint.Config.Dim, weights, true))(t)
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
