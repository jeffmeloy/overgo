package hostmath

import (
	"errors"
	"overgo/internal/checked"
)

// DepthwiseMemoryF64 filters frame-major x with independent channel kernels.
// Past taps are oldest-first and include the current frame; future taps begin
// one futureDilation after it. Cache contains at most the required preceding
// frames, oldest-first. Missing context is zero. The output is x plus each
// filtered direction, with FP64 dot products rounded to FP32 before addition.
// Destination must not overlap any input; errors may leave it partly written.
func DepthwiseMemoryF64(dst, x, past, future, cache []float32, frames, channels, pastDilation, futureDilation int) error {
	count, ok := checked.MulInt(frames, channels)
	if !ok || frames <= 0 || channels <= 0 || len(dst) != count || len(x) != count ||
		len(past) == 0 || len(past)%channels != 0 || len(future)%channels != 0 || len(cache)%channels != 0 || pastDilation <= 0 ||
		len(future) != 0 && futureDilation <= 0 {
		return errors.New("memory filter: invalid extent or dilation")
	}
	pastOrder, futureOrder := len(past)/channels, len(future)/channels
	context, ok := checked.MulInt(pastOrder-1, pastDilation)
	futureSpan, futureOK := checked.MulInt(futureOrder, futureDilation)
	if !ok || !futureOK || futureSpan < 0 || len(cache)/channels > context {
		return errors.New("memory filter: invalid context extent")
	}
	for _, source := range [][]float32{x, past, future, cache} {
		if checked.SlicesOverlap(dst, source) {
			return errors.New("memory filter: destination overlaps input")
		}
	}
	cachedFrames := len(cache) / channels
	for frame := range frames {
		for channel := range channels {
			var backward, forward float64
			for tap := range pastOrder {
				position := frame - (pastOrder-1-tap)*pastDilation
				var value float32
				if position >= 0 {
					value = x[position*channels+channel]
				} else if position >= -cachedFrames {
					value = cache[(cachedFrames+position)*channels+channel]
				}
				backward += float64(value) * float64(past[channel*pastOrder+tap])
			}
			for tap := range futureOrder {
				distance := (tap + 1) * futureDilation
				if distance < frames-frame {
					forward += float64(x[(frame+distance)*channels+channel]) * float64(future[channel*futureOrder+tap])
				}
			}
			index := frame*channels + channel
			dst[index] = x[index] + float32(backward)
			dst[index] += float32(forward)
			if !checked.Finite32(dst[index]) {
				return errors.New("memory filter: non-finite output")
			}
		}
	}
	return nil
}
