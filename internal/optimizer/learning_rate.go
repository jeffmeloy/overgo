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

// CLTMinSamples: the central-limit rule-of-thumb floor on effective samples
// the momentum EMA must average over (matches adaptive_new
// analysis.CLTMinSamples). A documented distributional assumption, not a
// free knob.
const CLTMinSamples = 30

// DeriveMomentum returns the Muon momentum whose EMA effective sample size
// (1+mu)/(1-mu) equals CLTMinSamples: mu = (N-1)/(N+1). With DeriveBaseLR
// this completes the minimal-hyperparameter contract — no tuned values.
func DeriveMomentum() float64 {
	return float64(CLTMinSamples-1) / float64(CLTMinSamples+1)
}
