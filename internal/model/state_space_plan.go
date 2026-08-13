package model

type recurrentMixerPolicy uint8

const (
	recurrentMixerNone recurrentMixerPolicy = iota
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

func (s Spec) compileRecurrentMixer(recurrent bool) recurrentMixerPolicy {
	profile := s.Profile()
	policy := recurrentMixerNone
	switch {
	case profile.Validation.Recurrent == RecurrentValidationMamba:
		policy = recurrentMixerSelectiveScan
	case profile.Validation.Recurrent == RecurrentValidationJamba && recurrent:
		policy = recurrentMixerWeightedSelectiveScan
	case profile.Validation.Recurrent == RecurrentValidationMamba2:
		policy = recurrentMixerGroupedSelectiveScan
	case profile.Validation.Recurrent == RecurrentValidationGraniteHybrid && recurrent:
		policy = recurrentMixerScaledGroupedSelectiveScan
	case profile.Validation.Recurrent == RecurrentValidationFalconH1:
		policy = recurrentMixerAttentionGroupedSelectiveScan
	case profile.Validation.Recurrent == RecurrentValidationPLaMo2 && recurrent:
		policy = recurrentMixerNormalizedSelectiveScan
	case profile.Validation.recurrentOneOf(RecurrentValidationNemotronH, RecurrentValidationNemotronHMoE):
		policy = recurrentMixerSparseGroupedSelectiveScan
	case profile.Attention == AttentionGatedDelta:
		policy = recurrentMixerGatedDelta
	case profile.Attention == AttentionLFM2 && recurrent:
		policy = recurrentMixerShortConvolution
	case profile.LayerTopology == LayerTopologyAffineWKV6:
		policy = recurrentMixerDynamicWKV6
	case profile.LayerTopology == LayerTopologyDynamicWKV6:
		policy = recurrentMixerAffineWKV6
	case profile.LayerTopology == LayerTopologyDynamicWKV7:
		policy = recurrentMixerDynamicWKV7
	case profile.Validation.MLA == MLAValidationKimiLinear:
		policy = recurrentMixerKeyedDelta
	}
	return policy
}
