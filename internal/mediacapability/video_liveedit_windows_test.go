//go:build windows

package mediacapability

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/pathidentity"
	"overgo/internal/safetensors"
)

// TestLiveEditRecordsItsOwnDirectory resolves a LiveEdit layout: the edit
// checkpoint in its directory and the Wan base beside it. The inventory
// roots at the parent the two share, and records the LiveEdit directory
// too, the one the capability resolves from, so the store can name it.
func TestLiveEditRecordsItsOwnDirectory(t *testing.T) {
	t.Parallel()
	parent, err := pathidentity.Canonical(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	edit, wan := filepath.Join(parent, "LiveEdit"), filepath.Join(parent, "Wan2.1-T2V-1.3B")
	checkpoint := func(path, value string) {
		t.Helper()
		if err := os.WriteFile(path, torchStateDict(t, value), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, directory := range []string{edit, wan} {
		if err := os.Mkdir(directory, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	checkpoint(filepath.Join(edit, "ar-forcing_002000.pt"), "\x00\x00\x80\x3f")
	checkpoint(filepath.Join(wan, "Wan2.1_VAE.pth"), "\x00\x00\x00\x40")
	if err := os.WriteFile(filepath.Join(wan, "config.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(wan, "diffusion_pytorch_model.safetensors"),
		map[string][]float32{"weight": {1}}, map[string][]int{"weight": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	source, err := resolveVideoSource(edit)
	if err != nil {
		t.Fatal(err)
	}
	model := source.Inventory.Manifest.ID
	for _, directory := range []string{parent, edit} {
		want := artifact.Location{Artifact: model, Kind: artifact.LocationDirectory, Value: directory}
		if !slices.Contains(source.Inventory.Locations, want) {
			t.Errorf("the LiveEdit inventory records no directory %s for %s: %+v", directory, model, source.Inventory.Locations)
		}
	}
}
