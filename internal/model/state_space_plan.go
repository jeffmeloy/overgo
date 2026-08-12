package model

type stateSpaceKind uint8

const (
	stateSpaceNone stateSpaceKind = iota
	stateSpaceMamba
	stateSpaceJamba
	stateSpaceMamba2
	stateSpaceGraniteHybrid
	stateSpaceFalconH1
	stateSpacePLaMo2
	stateSpaceNemotronH
	stateSpaceQwenGDN
	stateSpaceLFM2
	stateSpaceDynamicWKV6
	stateSpaceAffineWKV6
	stateSpaceDynamicWKV7
	stateSpaceKeyedDelta
)

// StateSpacePlan: recurrent graph/catalog contract.
type StateSpacePlan struct {
	kind      stateSpaceKind
	recurrent bool
	qwen      qwenGDNPolicy
}

func (s Spec) stateSpacePlan(layer uint32, recurrent bool) StateSpacePlan {
	profile := s.Profile()
	plan := StateSpacePlan{
		recurrent: recurrent || s.IsRecurrentLayer(layer),
		qwen:      profile.AttentionGraph.QwenGDN,
	}
	switch {
	case profile.Validation.Recurrent == RecurrentValidationMamba:
		plan.kind, plan.recurrent = stateSpaceMamba, true
	case profile.Validation.Recurrent == RecurrentValidationJamba && plan.recurrent:
		plan.kind = stateSpaceJamba
	case profile.Validation.Recurrent == RecurrentValidationMamba2:
		plan.kind, plan.recurrent = stateSpaceMamba2, true
	case profile.Validation.Recurrent == RecurrentValidationGraniteHybrid && plan.recurrent:
		plan.kind = stateSpaceGraniteHybrid
	case profile.Validation.Recurrent == RecurrentValidationFalconH1:
		plan.kind = stateSpaceFalconH1
	case profile.Validation.Recurrent == RecurrentValidationPLaMo2 && plan.recurrent:
		plan.kind = stateSpacePLaMo2
	case profile.Validation.recurrentOneOf(RecurrentValidationNemotronH, RecurrentValidationNemotronHMoE):
		plan.kind = stateSpaceNemotronH
	case profile.Attention == AttentionQwenGDN:
		plan.kind = stateSpaceQwenGDN
	case profile.Attention == AttentionLFM2 && plan.recurrent:
		plan.kind = stateSpaceLFM2
	case profile.DenseGraph == DenseGraphRWKV6Qwen2:
		plan.kind, plan.recurrent = stateSpaceDynamicWKV6, true
	case profile.DenseGraph == DenseGraphRWKV6:
		plan.kind, plan.recurrent = stateSpaceAffineWKV6, true
	case profile.DenseGraph == DenseGraphRWKV7:
		plan.kind, plan.recurrent = stateSpaceDynamicWKV7, true
	case profile.Validation.MLA == MLAValidationKimiLinear:
		plan.kind = stateSpaceKeyedDelta
	}
	return plan
}
