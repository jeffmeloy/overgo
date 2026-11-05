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
	base, err := LoadDenoiserConfig(modelDirectory, referenceDenoiserPolicy)
	testcheck.NoError(t, err)
	checkpointPath := filepath.Join(filepath.Dir(modelDirectory), "LiveEdit", "ar-forcing_002000.pt")
	checkpoint, err := CompileReferenceEditCheckpoint(checkpointPath, base)
	testcheck.NoError(t, err)
	if checkpoint.Config.InDim != 2*base.InDim || checkpoint.SourceChannels != base.InDim || checkpoint.Layers != base.NumLayers || len(checkpoint.bindings) != len(denoiserTensorLengths(checkpoint.Config))+len(projectionTensorNames) {
		t.Fatalf("checkpoint=%+v bindings=%d", checkpoint, len(checkpoint.bindings))
	}
	patch, ok := checkpoint.Binding("patch_embedding.weight")
	if !ok {
		t.Fatal("patch binding absent")
	}
	reader, err := pytorchzip.Open(checkpoint.Path)
	testcheck.NoError(t, err)
	defer reader.Close()
	values, err := reader.ReadBinding(patch)
	testcheck.NoError(t, err)
	if len(values) != checkpoint.Config.patchIn()*checkpoint.Config.Dim {
		t.Fatalf("patch values=%d want=%d", len(values), checkpoint.Config.patchIn()*checkpoint.Config.Dim)
	}
	t.Logf("LiveEdit bindings=%d input_channels=%d source_channels=%d layers=%d", len(checkpoint.bindings), checkpoint.Config.InDim, checkpoint.SourceChannels, checkpoint.Layers)
}
