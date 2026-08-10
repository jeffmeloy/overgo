package latentimage

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// ServingStatus: how far a recognized media artifact is toward being served.
type ServingStatus string

const (
	// StatusRecognized: arch recognized + geometry shape-derived + structurally
	// verified against the checkpoint; denoise/VAE-decode not yet ported. This
	// is the "servable-in-progress" state (serving-only, no training).
	StatusRecognized ServingStatus = "recognized (serving-in-progress)"
)

// MediaModel: one enumerated media artifact -- the discovery-facing view. The
// family tag lives here (discovery/recipe scope), the Spec's types stay
// functional. Kind is fixed to image diffusion for this pipeline.
type MediaModel struct {
	Dir     string
	Family  string
	Kind    string // "image-diffusion"
	Status  ServingStatus
	Serving bool // false: recognition only, no serving path yet
	Spec    *Spec
}

// mediaKind: the modality this executor serves.
const mediaKind = "image-diffusion"

// Enumerate scans the immediate subdirectories of root and returns a
// MediaModel for each that RecognizePipeline classifies. Directories that are
// not this pipeline are skipped silently (discovery may probe anything); a
// malformed matching artifact is a hard error. Results are dir-sorted.
func Enumerate(root string) ([]MediaModel, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("latentimage: enumerate %s: %w", root, err)
	}
	var models []MediaModel
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		dir := filepath.Join(root, entry.Name())
		spec, ok, err := RecognizePipeline(dir)
		if err != nil {
			return nil, fmt.Errorf("latentimage: enumerate %s: %w", entry.Name(), err)
		}
		if !ok {
			continue
		}
		models = append(models, MediaModel{
			Dir:     dir,
			Family:  spec.Family,
			Kind:    mediaKind,
			Status:  StatusRecognized,
			Serving: false,
			Spec:    spec,
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Dir < models[j].Dir })
	return models, nil
}
