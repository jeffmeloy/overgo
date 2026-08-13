package model

type recurrentCadence uint8

const (
	recurrentCadenceExplicit recurrentCadence = iota
	recurrentCadenceAttentionInterval
)

type moeCadence uint8

const (
	moeCadenceNone moeCadence = iota
	moeCadenceOffsetOne
	moeCadenceEvery
	moeCadenceAfterDense
)

type slidingCadence uint8

const (
	slidingCadenceNone slidingCadence = iota
	slidingCadenceNonRecurrent
	slidingCadenceExceptFirst
	slidingCadenceExceptLast
)

// LayerCadencePolicy: recurrent, expert, and sliding layer schedules.
type LayerCadencePolicy struct {
	Recurrent             recurrentCadence
	MoE                   moeCadence
	Sliding               slidingCadence
	FullIndexerEveryLayer bool
	FullIndexerContext    uint32
	FullIndexerPrefix     uint32
	FullIndexerPeriod     uint32
}

func (p LayerCadencePolicy) fullIndexer(context, layer uint32) bool {
	if p.FullIndexerEveryLayer || p.FullIndexerContext > 0 && context < p.FullIndexerContext {
		return true
	}
	return p.FullIndexerPeriod > 0 && (layer < p.FullIndexerPrefix ||
		(layer-p.FullIndexerPrefix)%p.FullIndexerPeriod == 0)
}

func (p LayerCadencePolicy) recurrent(spec Spec, block uint32) bool {
	if block >= spec.BlockCount {
		return false
	}
	if len(spec.RecurrentLayers) == int(spec.BlockCount) {
		return spec.RecurrentLayers[block]
	}
	return p.Recurrent == recurrentCadenceAttentionInterval &&
		spec.FullAttentionInterval > 0 && (block+1)%spec.FullAttentionInterval != 0
}

func (p LayerCadencePolicy) moe(spec Spec, block uint32) bool {
	if block >= spec.BlockCount {
		return false
	}
	switch p.MoE {
	case moeCadenceOffsetOne:
		return spec.MoELayerStep > 1 && block%spec.MoELayerStep == 1
	case moeCadenceEvery:
		return spec.MoELayerStep > 0 && (block+1)%spec.MoELayerStep == 0
	case moeCadenceAfterDense:
		return block >= spec.LeadingDenseBlocks && spec.MoELayerStep > 0 &&
			(block+1)%spec.MoELayerStep == 0
	default:
		return false
	}
}

func (p LayerCadencePolicy) sliding(spec Spec, block uint32) bool {
	if block >= spec.BlockCount {
		return false
	}
	switch p.Sliding {
	case slidingCadenceNonRecurrent:
		return spec.SlidingWindow > 0 && !p.recurrent(spec, block)
	case slidingCadenceExceptFirst:
		return spec.SlidingWindow > 0 && spec.SlidingPattern > 0 && block%spec.SlidingPattern != 0
	}
	if block < uint32(len(spec.SlidingLayers)) {
		return spec.SlidingLayers[block]
	}
	return p.Sliding == slidingCadenceExceptLast && spec.SlidingWindow > 0 && spec.SlidingPattern > 0 &&
		block%spec.SlidingPattern < spec.SlidingPattern-1
}
