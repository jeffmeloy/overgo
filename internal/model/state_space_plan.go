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
)

// StateSpacePlan: recurrent graph/catalog contract.
type StateSpacePlan struct {
	kind      stateSpaceKind
	recurrent bool
	moe       bool
	qwen      qwenGDNPolicy
}

func (s Spec) stateSpacePlan(layer uint32) StateSpacePlan {
	profile := s.Profile()
	plan := StateSpacePlan{
		recurrent: s.IsRecurrentLayer(layer), moe: profile.Has(ArchitectureMoE),
		qwen: profile.AttentionGraph.QwenGDN,
	}
	switch {
	case profile.Block == BlockMamba:
		plan.kind, plan.recurrent = stateSpaceMamba, true
	case profile.RecurrentBlock == BlockJamba && plan.recurrent:
		plan.kind = stateSpaceJamba
	case profile.Block == BlockMamba2:
		plan.kind, plan.recurrent = stateSpaceMamba2, true
	case profile.RecurrentBlock == BlockGraniteHybrid && plan.recurrent:
		plan.kind = stateSpaceGraniteHybrid
	case profile.Block == BlockFalconH1:
		plan.kind = stateSpaceFalconH1
	case profile.RecurrentBlock == BlockPLaMo2 && plan.recurrent:
		plan.kind = stateSpacePLaMo2
	case profile.Block == BlockNemotronH:
		plan.kind = stateSpaceNemotronH
	case profile.Attention == AttentionQwenGDN:
		plan.kind = stateSpaceQwenGDN
	case profile.Attention == AttentionLFM2 && plan.recurrent:
		plan.kind = stateSpaceLFM2
	}
	return plan
}

func (p StateSpacePlan) mamba2Mixer() bool {
	return p.kind == stateSpaceMamba2 || p.kind == stateSpaceGraniteHybrid ||
		p.kind == stateSpaceFalconH1 || p.kind == stateSpaceNemotronH
}
