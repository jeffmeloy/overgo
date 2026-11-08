package latentvideo

import (
	"fmt"
	"slices"
	"strings"

	"overgo/internal/pytorchzip"
	"overgo/internal/safetensors"
)

// selfForcingPrefix opens every tensor name of a Self-Forcing checkpoint: its trainer saves the generator as "model".
const selfForcingPrefix = "model."

// matchesWanDiT says a Self-Forcing student checkpoint is a whole Wan DiT, the one in directory distilled: each of its
// tensors, its trainer's prefix aside, is one of that DiT's, of the same shape, and it holds every one of them. A
// student is served beside its teacher on the teacher's text encoder and autoencoder, so a tensor of another shape or
// a missing one is refused before anything loads.
func matchesWanDiT(checkpoint, directory string) error {
	catalog, err := pytorchzip.ReadCatalog(checkpoint)
	if err != nil {
		return fmt.Errorf("self-forcing: %w", err)
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return fmt.Errorf("self-forcing: %w", err)
	}
	defer source.Close()
	shapes, err := source.IntShapes()
	if err != nil {
		return err
	}
	if len(catalog.Tensors) != len(shapes) {
		return fmt.Errorf("self-forcing: %s holds %d tensors, and the Wan DiT in %s holds %d", checkpoint, len(catalog.Tensors), directory, len(shapes))
	}
	for _, meta := range catalog.Tensors {
		name, prefixed := strings.CutPrefix(meta.Name, selfForcingPrefix)
		want, held := shapes[name]
		if !prefixed || !held || !slices.EqualFunc(meta.Shape, want, func(have int64, wanted int) bool { return have == int64(wanted) }) {
			return fmt.Errorf("self-forcing: %s's tensor %s %v is no tensor of the Wan DiT in %s (%v)", checkpoint, meta.Name, meta.Shape, directory, want)
		}
	}
	return nil
}
