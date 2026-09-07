package media

import (
	"bytes"
	"errors"
	"image/gif"
)

// GIFShape is what a recorded GIF says about itself: its frame count and
// the first frame's bounds, read from the file rather than a JSON copy.
type GIFShape struct {
	Frames int
	Height int
	Width  int
}

// DecodeGIFShape reads the frame count and bounds of an encoded GIF; an
// animation with no frames is refused.
func DecodeGIFShape(data []byte) (GIFShape, error) {
	decoded, err := gif.DecodeAll(bytes.NewReader(data))
	if err != nil {
		return GIFShape{}, err
	}
	if len(decoded.Image) == 0 {
		return GIFShape{}, errors.New("media: GIF has no frames")
	}
	bounds := decoded.Image[0].Bounds()
	return GIFShape{Frames: len(decoded.Image), Height: bounds.Dy(), Width: bounds.Dx()}, nil
}
