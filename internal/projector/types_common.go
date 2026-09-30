package projector

import (
	"errors"

	"overgo/internal/tensor/reference"
)

var errRunnerClosed = errors.New("projector: runner is closed")

type visionActivation uint8

const (
	visionQuickGELU visionActivation = iota
	visionGELU
	visionSiLU
)

type gridImage struct {
	PixelValues []float32
	GridH       int
	GridW       int
}

// mediaPixelBudget defines validated raster area bounds.
type mediaPixelBudget struct {
	MinPixels int `json:"min_pixels"`
	MaxPixels int `json:"max_pixels"`
}

type pixelBudget = mediaPixelBudget
type rasterPatchOptions = mediaPixelBudget

type mediaPreprocessOwner struct{ preprocess MediaPreprocessProfile }

func (owner *mediaPreprocessOwner) setMediaPreprocess(profile MediaPreprocessProfile) {
	owner.preprocess = profile
}

type rasterPatchImage = gridImage

type gridOutput struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
	MergeSize  int
}
