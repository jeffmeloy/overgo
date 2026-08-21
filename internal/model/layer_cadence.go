package model

import "overgo/internal/tensor"

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
	if p.FullIndexerEveryLayer ||
		p.FullIndexerContext > tensor.FirstOffset && context < p.FullIndexerContext {
		return true
	}
	return p.FullIndexerPeriod > tensor.FirstOffset && (layer < p.FullIndexerPrefix ||
		(layer-p.FullIndexerPrefix)%p.FullIndexerPeriod == tensor.FirstOffset)
}

func (p LayerCadencePolicy) recurrent(spec Spec, block uint32) bool {
	if block >= spec.BlockCount {
		return false
	}
	if len(spec.RecurrentLayers) == int(spec.BlockCount) {
		return spec.RecurrentLayers[block]
	}
	return p.Recurrent == recurrentCadenceAttentionInterval &&
		spec.FullAttentionInterval > tensor.FirstOffset &&
		(block+tensor.SingletonExtent)%spec.FullAttentionInterval != tensor.FirstOffset
}

func (p LayerCadencePolicy) moe(spec Spec, block uint32) bool {
	if block >= spec.BlockCount {
		return false
	}
	switch p.MoE {
	case moeCadenceOffsetOne:
		return spec.MoELayerStep > tensor.SingletonExtent &&
			block%spec.MoELayerStep == tensor.SingletonExtent
	case moeCadenceEvery:
		return spec.MoELayerStep > tensor.FirstOffset &&
			(block+tensor.SingletonExtent)%spec.MoELayerStep == tensor.FirstOffset
	case moeCadenceAfterDense:
		return block >= spec.LeadingDenseBlocks && spec.MoELayerStep > tensor.FirstOffset &&
			(block+tensor.SingletonExtent)%spec.MoELayerStep == tensor.FirstOffset
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
		return spec.SlidingWindow > tensor.FirstOffset && !p.recurrent(spec, block)
	case slidingCadenceExceptFirst:
		return spec.SlidingWindow > tensor.FirstOffset && spec.SlidingPattern > tensor.FirstOffset &&
			block%spec.SlidingPattern != tensor.FirstOffset
	}
	if block < uint32(len(spec.SlidingLayers)) {
		return spec.SlidingLayers[block]
	}
	return p.Sliding == slidingCadenceExceptLast && spec.SlidingWindow > tensor.FirstOffset &&
		spec.SlidingPattern > tensor.FirstOffset &&
		block%spec.SlidingPattern < spec.SlidingPattern-tensor.SingletonExtent
}

func (p LayerCadencePolicy) periodicRescale(spec Spec, block uint32) bool {
	return spec.RescaleEvery > tensor.FirstOffset &&
		(block+tensor.SingletonExtent)%spec.RescaleEvery == tensor.FirstOffset
}
