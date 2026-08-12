package optimizer

import "math"

// BaseLRParamExponent: the Muon base learning rate scales as n_params^p with
// p = -1/2 -- the displacement-budget scaling under which one derived rate holds
// across model scale (matches adaptive_new config.ConfigBaseLRParamExponent). A
// documented distributional assumption, not a free knob.
const BaseLRParamExponent = -0.5

// DeriveBaseLR returns the scale-derived Muon base learning rate for a model of
// nParams trainable parameters: nParams^BaseLRParamExponent. Non-positive
// nParams yields 0 so the caller fails loudly rather than training at a silent
// default.
func DeriveBaseLR(nParams int) float64 {
	if nParams <= 0 {
		return 0
	}
	return math.Pow(float64(nParams), BaseLRParamExponent)
}
