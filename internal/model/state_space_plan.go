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

func compileRecurrentMixer(profile ArchitectureProfile, recurrent bool) RecurrentMixerPolicy {
	policy := profile.RecurrentMixer
	if !recurrent && policy.recurrentOnly() {
		return recurrentMixerNone
	}
	return policy
}
