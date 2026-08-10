package thoughtbank

import "math/rand"

// referenceRMSNormEps is the reference RMSNorm epsilon shared by the ported
// fixtures (== DefaultNormEps).
const referenceRMSNormEps = DefaultNormEps

// randSlice fills a deterministic small-magnitude weight vector.
func randSlice(rng *rand.Rand, n int, scale float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((rng.Float64()*2 - 1) * scale)
	}
	return v
}

// randVec is an alias kept for the ported fixture builders.
func randVec(rng *rand.Rand, n int, scale float64) []float32 { return randSlice(rng, n, scale) }

// argmaxF32 returns the index of the largest element (first on ties).
func argmaxF32(v []float32) int {
	best, at := v[0], 0
	for i, x := range v {
		if x > best {
			best, at = x, i
		}
	}
	return at
}
