package media

import "overgo/internal/tensor"

// ResamplePoly resamples a signal by up/down with precomputed odd-length FIR
// taps centered on the middle tap. The result length is ceil(len(x)*up/down).
func ResamplePoly(x []float32, up, down int, taps []float64) []float32 {
	if up <= tensor.FirstOffset || down <= tensor.FirstOffset || len(taps) == tensor.FirstOffset {
		return nil
	}
	halfLen := (len(taps) - tensor.SingletonExtent) / tensor.PairedExtent
	nOut := (len(x)*up + down - tensor.SingletonExtent) / down
	out := make([]float32, nOut)
	for i := range out {
		center := i*down + halfLen
		var acc float64
		for j := range taps {
			u := center - j
			if u < tensor.FirstOffset {
				break
			}
			if u%up != tensor.FirstOffset {
				continue
			}
			xi := u / up
			if xi < len(x) {
				acc += taps[j] * float64(x[xi])
			}
		}
		out[i] = float32(acc)
	}
	return out
}
