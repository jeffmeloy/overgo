package optimizer

import "math"

// BaseLRParamExponent: displacement-budget scale exponent.
const BaseLRParamExponent = -0.5

// DeriveBaseLR: parameter-count-derived Muon rate.
func DeriveBaseLR(nParams int) float64 {
	if nParams <= 0 {
		return 0
	}
	return math.Pow(float64(nParams), BaseLRParamExponent)
}

// CLTMinSamples: declared EMA effective-sample floor.
const CLTMinSamples = 30

// DeriveMomentum: CLTMinSamples-equivalent EMA momentum.
func DeriveMomentum() float64 {
	return float64(CLTMinSamples-1) / float64(CLTMinSamples+1)
}
