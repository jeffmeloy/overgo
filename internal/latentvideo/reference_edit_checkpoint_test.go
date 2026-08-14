package latentvideo

import (
	"path/filepath"
	"testing"

	"overgo/internal/pytorchzip"
)

func TestLiveEditCheckpointBinding(t *testing.T) {
	modelDirectory := wanModelDir(t)
	base, err := LoadDenoiserConfig(modelDirectory, referenceDenoiserPolicy)
	if err != nil {
		t.Fatal(err)
	}
	checkpointPath := filepath.Join(filepath.Dir(modelDirectory), "LiveEdit", "ar-forcing_002000.pt")
	checkpoint, err := CompileReferenceEditCheckpoint(checkpointPath, base)
	if err != nil {
		t.Fatal(err)
	}
	if checkpoint.Config.InDim != 2*base.InDim || checkpoint.SourceChannels != base.InDim || checkpoint.Layers != base.NumLayers || len(checkpoint.bindings) != len(denoiserTensorLengths(checkpoint.Config)) {
		t.Fatalf("checkpoint=%+v bindings=%d", checkpoint, len(checkpoint.bindings))
	}
	patch, ok := checkpoint.Binding("patch_embedding.weight")
	if !ok {
		t.Fatal("patch binding absent")
	}
	reader, err := pytorchzip.Open(checkpoint.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	values, err := reader.ReadBinding(patch)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != checkpoint.Config.patchIn()*checkpoint.Config.Dim {
		t.Fatalf("patch values=%d want=%d", len(values), checkpoint.Config.patchIn()*checkpoint.Config.Dim)
	}
	t.Logf("LiveEdit bindings=%d input_channels=%d source_channels=%d layers=%d", len(checkpoint.bindings), checkpoint.Config.InDim, checkpoint.SourceChannels, checkpoint.Layers)
}
