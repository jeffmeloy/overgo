package media

import (
	"context"
	"encoding/binary"
	"errors"
	"math"

	"overgo/internal/checked"
)

// ResamplePoly resamples x by up/down using explicit odd-length FIR taps
// centered on the middle tap. The output length is ceil(len(x)*up/down), with
// float64 accumulation stored as float32. It reuses destination capacity and
// rejects overlapping input/output storage, non-finite values and overflow.
// Errors return no output; cancellation may overwrite destination scratch.
func ResamplePoly(ctx context.Context, destination, x []float32, up, down int, taps []float64) ([]float32, error) {
	if ctx == nil || up <= 0 || down <= 0 || len(taps) == 0 || len(taps)%2 != 1 {
		return nil, errors.New("audio resample: invalid context, ratio or FIR taps")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, value := range x {
		if !checked.Finite32(value) {
			return nil, errors.New("audio resample: non-finite input")
		}
	}
	for _, tap := range taps {
		if !checked.Finite64(tap) {
			return nil, errors.New("audio resample: non-finite tap")
		}
	}
	product, ok := checked.MulInt(len(x), up)
	if !ok || product > math.MaxInt-(down-1) {
		return nil, errors.New("audio resample: output extent overflows")
	}
	count := (product + down - 1) / down
	half := (len(taps) - 1) / 2
	if count > math.MaxInt/binary.Size(float32(0)) {
		return nil, errors.New("audio resample: output bytes overflow")
	}
	if count != 0 {
		last, ok := checked.MulInt(count-1, down)
		if !ok || last > math.MaxInt-half {
			return nil, errors.New("audio resample: FIR index overflows")
		}
	}
	if cap(destination) < count {
		destination = make([]float32, count)
	} else {
		destination = destination[:count]
	}
	if checked.SlicesOverlap(destination, x) {
		return nil, errors.New("audio resample: destination overlaps input")
	}
	for index := range destination {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		center := index*down + half
		var sum float64
		for tapIndex, tap := range taps {
			u := center - tapIndex
			if u < 0 {
				break
			}
			if u%up != 0 {
				continue
			}
			input := u / up
			if input < len(x) {
				sum += tap * float64(x[input])
			}
		}
		destination[index] = float32(sum)
		if !checked.Finite32(destination[index]) {
			return nil, errors.New("audio resample: non-finite output")
		}
	}
	return destination, nil
}
