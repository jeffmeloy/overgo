package projector

import "llamacpp2go/internal/tensor/reference"

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

type gridOutput struct {
	Embeddings reference.Value
	GridH      int
	GridW      int
	MergeSize  int
}
