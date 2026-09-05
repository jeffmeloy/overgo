package audiodsp

import (
	"context"
	"errors"

	"overgo/internal/checked"
)

// AppendFrameDeltas writes frame-major [static, delta] features into dst.
// Source has bands values per frame; dst must have twice its length and must
// not overlap it. The radius declares a centered regression window, with edge
// replication and denominator 2*sum(n*n), n=1..radius. Float64 accumulation is
// rounded once to float32. Consecutive frames may then be stacked by reshaping
// this frame-major output, without a transpose or another copy.
// Cancellation may leave dst partially written; source is never modified.
func AppendFrameDeltas(ctx context.Context, dst, source []float32, bands, radius int) error {
	count, ok := checked.MulInt(len(source), 2)
	if ctx == nil || !ok || bands <= 0 || radius <= 0 || len(source) == 0 ||
		len(source)%bands != 0 || len(dst) != count || checked.SlicesOverlap(dst, source) {
		return errors.New("audio deltas: invalid shape, radius or overlapping storage")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	for _, value := range source {
		if !checked.Finite32(value) {
			return errors.New("audio deltas: non-finite input")
		}
	}
	frames := len(source) / bands
	r := float64(radius)
	denominator := r * (r + 1) * (2*r + 1) / 3
	// Beyond frames-1 both sides always select the replicated endpoints.
	// Sum that constant tail analytically instead of visiting an unbounded
	// declared window, and avoid overflowing integer frame+radius indices.
	limit := min(radius, frames-1)
	l := float64(limit)
	tail := (r*(r+1) - l*(l+1)) / 2
	for frame := range frames {
		if err := ctx.Err(); err != nil {
			return err
		}
		output := dst[frame*2*bands : (frame+1)*2*bands]
		copy(output, source[frame*bands:(frame+1)*bands])
		for band := range bands {
			var sum float64
			for offset := 1; offset <= limit; offset++ {
				left := (frame-min(offset, frame))*bands + band
				right := (frame+min(offset, frames-1-frame))*bands + band
				sum += float64(offset) * (float64(source[right]) - float64(source[left]))
			}
			sum += tail * (float64(source[(frames-1)*bands+band]) - float64(source[band]))
			output[bands+band] = float32(sum / denominator)
		}
	}
	return nil
}
