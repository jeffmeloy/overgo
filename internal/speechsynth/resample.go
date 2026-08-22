// Polyphase resampling (scipy.signal.resample_poly parity, ladder g9).
// INPUT-side stage only: the reference emits WAV at the codec sample rate
// with no output resampling; this prepares voice/clip audio recorded at
// other rates (e.g. 16 kHz -> 24 kHz with up=3, down=2).
package speechsynth

import "overgo/internal/media"

// ResamplePoly resamples x by up/down with precomputed odd-length FIR taps
// centered at (len(taps)-1)/2 (firwin low-pass scaled by up; the fixture
// carries the exact taps). Output length ceil(len(x)*up/down); f64
// accumulation stored f32.
func ResamplePoly(x []float32, up, down int, taps []float64) []float32 {
	return media.ResamplePoly(x, up, down, taps)
}
