package model

import (
	"reflect"
	"slices"

	"overgo/internal/tensor"
)

// ForwardOperation: neutral top-level execution entry.
type ForwardOperation uint8

const (
	ForwardOperationCached ForwardOperation = iota
	ForwardOperationBidirectional
	ForwardOperationAudioTokens
	ForwardOperationEncoder
	ForwardOperationSession
	ForwardOperationDiffusion
	forwardOperationCount
)

// ForwardSession: required coordinator contract.
type ForwardSession uint8

const (
	ForwardSessionNone ForwardSession = iota
	ForwardSessionPairedFeatures
	ForwardSessionFeatureDraft
	ForwardSessionPairedProjection
	ForwardSessionEncoderDecoder
	forwardSessionCount
)

// AlternateStateProgram: compiled alternate-state execution facts.
type AlternateStateProgram struct {
	StateCount            uint32
	ActiveState           uint32
	SparseLayerCount      uint32
	SparsityStdMultiplier float32
	EmbeddingLength       uint32
	NormalizationEpsilon  float32
	enabled               bool
}

// SparseLayer reports thresholded gated activation for layer.
func (p AlternateStateProgram) SparseLayer(layer uint32) bool {
	return layer < p.SparseLayerCount
}

// ForwardProgram: compiled top-level execution contract.
type ForwardProgram struct {
	Operation               ForwardOperation
	Session                 ForwardSession
	Waveform                AudioWaveformPlan
	SequenceResiduals       []sequenceResidualOperator
	SequenceInputKernel     uint32
	SequenceResidualKernel  uint32
	SequenceDepthwiseKernel uint32
	Alternate               AlternateStateProgram
	continuousBatch         bool
	persistentDeviceCache   bool
	layerCapture            bool
	deviceBatchSelection    bool
	nonCausalRecurrent      bool
}

func compileForwardProgram(spec Spec, profile ArchitectureProfile) ForwardProgram {
	forward := resolveForwardProgram(profile.Forward.clone(), spec.NonCausalAttention)
	cached := forward.Operation == ForwardOperationCached
	if profile.LayerTopology == LayerTopologySplitProjection {
		forward.Alternate = AlternateStateProgram{
			enabled: true, StateCount: spec.AltUpCount, ActiveState: spec.AltUpActive,
			SparseLayerCount:      spec.SparseLayerCount,
			SparsityStdMultiplier: spec.SparsityStdMultiplier,
			EmbeddingLength:       spec.EmbeddingLength,
			NormalizationEpsilon:  spec.RMSNormEpsilon,
		}
	}
	forward.continuousBatch = cached
	forward.persistentDeviceCache = cached && !forward.AlternateStates() && !profile.Has(ArchitectureLatent)
	forward.layerCapture = cached && !forward.AlternateStates()
	forward.deviceBatchSelection = profile.Attention == AttentionGatedDelta
	forward.nonCausalRecurrent = profile.Attention == AttentionShortConvolution
	return forward
}

func (p ForwardProgram) clone() ForwardProgram {
	p.SequenceResiduals = slices.Clone(p.SequenceResiduals)
	return p
}

func (p ForwardProgram) equal(other ForwardProgram) bool {
	if !slices.Equal(p.SequenceResiduals, other.SequenceResiduals) {
		return false
	}
	p.SequenceResiduals, other.SequenceResiduals = nil, nil
	return reflect.DeepEqual(p, other)
}

func (p ForwardProgram) ContinuousBatch() bool       { return p.continuousBatch }
func (p ForwardProgram) PersistentDeviceCache() bool { return p.persistentDeviceCache }
func (p ForwardProgram) LayerCapture() bool          { return p.layerCapture }
func (p ForwardProgram) DeviceBatchSelection() bool  { return p.deviceBatchSelection }
func (p ForwardProgram) NonCausalRecurrent() bool    { return p.nonCausalRecurrent }
func (p ForwardProgram) AlternateStates() bool       { return p.Alternate.enabled }

func (p ForwardProgram) valid() bool {
	if p.Operation >= forwardOperationCount || p.Session >= forwardSessionCount {
		return false
	}
	audio := p.Operation == ForwardOperationAudioTokens
	sequenceKernels := p.SequenceInputKernel > tensor.FirstOffset &&
		p.SequenceResidualKernel > tensor.FirstOffset && p.SequenceDepthwiseKernel > tensor.FirstOffset
	if audio != p.Waveform.Valid() || audio != (len(p.SequenceResiduals) > 0) || audio != sequenceKernels {
		return false
	}
	for _, operator := range p.SequenceResiduals {
		if operator >= sequenceResidualOperatorCount {
			return false
		}
	}
	return (p.Operation == ForwardOperationSession) == (p.Session != ForwardSessionNone)
}

func resolveForwardProgram(program ForwardProgram, nonCausal bool) ForwardProgram {
	if program.Operation == ForwardOperationCached && nonCausal {
		program.Operation = ForwardOperationBidirectional
	}
	return program
}
