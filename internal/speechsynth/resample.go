// Polyphase resampling (scipy.signal.resample_poly parity, ladder g9).
// INPUT-side stage only: the reference emits WAV at the codec sample rate
// with no output resampling; this prepares voice/clip audio recorded at
// other rates (e.g. 16 kHz -> 24 kHz with up=3, down=2).
package speechsynth

// ResamplePoly resamples x by up/down with precomputed odd-length FIR taps
// centered at (len(taps)-1)/2 (firwin low-pass scaled by up; the fixture
// carries the exact taps). Output length ceil(len(x)*up/down); f64
// accumulation stored f32.
func ResamplePoly(x []float32, up, down int, taps []float64) []float32 {
	if up <= 0 || down <= 0 || len(taps) == 0 {
		return nil
	}
	halfLen := (len(taps) - 1) / 2
	nOut := (len(x)*up + down - 1) / down
	out := make([]float32, nOut)
	for i := range out {
		center := i*down + halfLen
		var acc float64
		for j := range taps {
			u := center - j
			if u < 0 {
				break
			}
			if u%up != 0 {
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
