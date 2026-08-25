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

// MediaPixelBudget defines validated raster area bounds.
type MediaPixelBudget struct {
	MinPixels int `json:"min_pixels"`
	MaxPixels int `json:"max_pixels"`
}

type pixelBudget = MediaPixelBudget
type RasterPatchOptions = MediaPixelBudget

type mediaPreprocessOwner struct{ preprocess MediaPreprocessProfile }

func (owner *mediaPreprocessOwner) setMediaPreprocess(profile MediaPreprocessProfile) {
	owner.preprocess = profile
}

type RasterPatchImage = gridImage

type gridOutput struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
	MergeSize  int
}
