package model

// RecurrentMixerPolicy: compiled recurrent-mixing mathematics.
type RecurrentMixerPolicy uint8

const (
	recurrentMixerNone RecurrentMixerPolicy = iota
	recurrentMixerSelectiveScan
	recurrentMixerWeightedSelectiveScan
	recurrentMixerGroupedSelectiveScan
	recurrentMixerScaledGroupedSelectiveScan
	recurrentMixerAttentionGroupedSelectiveScan
	recurrentMixerNormalizedSelectiveScan
	recurrentMixerSparseGroupedSelectiveScan
	recurrentMixerGatedDelta
	recurrentMixerShortConvolution
	recurrentMixerDynamicWKV6
	recurrentMixerAffineWKV6
	recurrentMixerDynamicWKV7
	recurrentMixerKeyedDelta
)

func (p RecurrentMixerPolicy) recurrentOnly() bool {
	return p == recurrentMixerWeightedSelectiveScan || p == recurrentMixerScaledGroupedSelectiveScan ||
		p == recurrentMixerNormalizedSelectiveScan || p == recurrentMixerShortConvolution
}

func (s Spec) compileRecurrentMixer(recurrent bool) RecurrentMixerPolicy {
	policy := s.Profile().RecurrentMixer
	if !recurrent && policy.recurrentOnly() {
		return recurrentMixerNone
	}
	return policy
}
