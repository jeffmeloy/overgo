package media

import "overgo/internal/tensor/dtype"

// NoiseMix carries signal/noise mixing scalars for latent initialization.
type NoiseMix struct {
	SignalWeight float32
	NoiseWeight  float32
	NoiseScale   float32
}

// FreshNoiseMix returns the pure-noise initialization contract.
func FreshNoiseMix() NoiseMix {
	return NoiseMix{NoiseWeight: 1, NoiseScale: 1}
}

// MixNoiseBF16Into applies an elementwise signal/noise mix with bfloat16
// rounding at both the scaled-noise and output storage boundaries.
func MixNoiseBF16Into(destination, noise []float32, mix NoiseMix) {
	for index := range noise {
		scaledNoise := dtype.RoundBF16(noise[index] * mix.NoiseScale)
		destination[index] = dtype.RoundBF16(mix.SignalWeight*destination[index] + mix.NoiseWeight*scaledNoise)
	}
}
