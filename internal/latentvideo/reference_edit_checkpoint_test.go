//go:build integration

package latentvideo

import (
	"path/filepath"
	"testing"

	"overgo/internal/pytorchzip"
	"overgo/internal/testcheck"
)

func TestLiveEditCheckpointBinding(t *testing.T) {
	modelDirectory := wanModelDir(t)
	base := testcheck.Must(LoadDenoiserConfig(modelDirectory, referenceDenoiserPolicy))(t)
	checkpointPath := filepath.Join(filepath.Dir(modelDirectory), "LiveEdit", "ar-forcing_002000.pt")
	checkpoint := testcheck.Must(CompileReferenceEditCheckpoint(checkpointPath, base))(t)
	if checkpoint.Config.InDim != 2*base.InDim || checkpoint.SourceChannels != base.InDim || checkpoint.Layers != base.NumLayers || len(checkpoint.bindings) != len(denoiserTensorLengths(checkpoint.Config))+len(projectionTensorNames) {
		t.Fatalf("checkpoint=%+v bindings=%d", checkpoint, len(checkpoint.bindings))
	}
	patch, ok := checkpoint.Binding("patch_embedding.weight")
	if !ok {
		t.Fatal("patch binding absent")
	}
	reader := testcheck.Must(pytorchzip.Open(checkpoint.Path))(t)
	defer reader.Close()
	values := testcheck.Must(reader.ReadBinding(patch))(t)
	if len(values) != checkpoint.Config.patchIn()*checkpoint.Config.Dim {
		t.Fatalf("patch values=%d want=%d", len(values), checkpoint.Config.patchIn()*checkpoint.Config.Dim)
	}
	t.Logf("LiveEdit bindings=%d input_channels=%d source_channels=%d layers=%d", len(checkpoint.bindings), checkpoint.Config.InDim, checkpoint.SourceChannels, checkpoint.Layers)
}
