package media

import (
	"bytes"
	"errors"
	"fmt"
	"image/gif"
	"math"
)

// GIFShape is what a recorded GIF says about itself: its frame count, the
// first frame's bounds and its frame rate, read from the file rather than a
// JSON copy.
type GIFShape struct {
	Frames int
	Height int
	Width  int
	FPS    int
}

// DecodeGIFShape reads the frame count, bounds and frame rate of an encoded
// GIF; an animation with no frames is refused.
//   - FPS = the rate the frame delays show, to the nearest whole frame a second
//   - nearest because GIFFrameDelay rounds each frame boundary to a centisecond, so the delays sum to within half a centisecond of the clip's length
//   - a clip whose delays show no rate of at least one frame a second is refused, because a published clip carries a positive rate.
func DecodeGIFShape(data []byte) (GIFShape, error) {
	decoded, err := gif.DecodeAll(bytes.NewReader(data))
	if err != nil {
		return GIFShape{}, err
	}
	if len(decoded.Image) == 0 {
		return GIFShape{}, errors.New("media: GIF has no frames")
	}
	shown := 0
	for _, delay := range decoded.Delay {
		shown += delay
	}
	frames := len(decoded.Image)
	fps := 0
	if shown > 0 {
		fps = int(math.Round(float64(frames*centisecondsPerSecond) / float64(shown)))
	}
	if fps <= 0 {
		return GIFShape{}, fmt.Errorf("media: a GIF of %d frames shown for %d centiseconds has no frame rate of at least one a second", frames, shown)
	}
	bounds := decoded.Image[0].Bounds()
	return GIFShape{Frames: frames, Height: bounds.Dy(), Width: bounds.Dx(), FPS: fps}, nil
}
