package projector

import (
	"errors"

	"overgo/internal/tensor/reference"
)

var errRunnerClosed = errors.New("projector: runner is closed")

type gridImage struct {
	PixelValues []float32
	GridH       int
	GridW       int
}

type pixelBudget struct {
	MinPixels      int
	MaxPixels      int
	MaxAspectRatio int
}

type RasterPatchOptions = pixelBudget
type RasterPatchImage = gridImage

type gridOutput struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
	MergeSize  int
}
