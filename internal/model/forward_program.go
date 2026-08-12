package model

// ForwardOperation: neutral top-level execution entry.
type ForwardOperation uint8

const (
	ForwardOperationCached ForwardOperation = iota
	ForwardOperationBidirectional
	ForwardOperationAudioTokens
	ForwardOperationEncoder
	ForwardOperationSession
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

// ForwardProgram: compiled top-level execution contract.
type ForwardProgram struct {
	Operation ForwardOperation
	Session   ForwardSession
}

func (p ForwardProgram) valid() bool {
	if p.Operation >= forwardOperationCount || p.Session >= forwardSessionCount {
		return false
	}
	return (p.Operation == ForwardOperationSession) == (p.Session != ForwardSessionNone)
}

func resolveForwardProgram(program ForwardProgram, nonCausal bool) ForwardProgram {
	if program.Operation == ForwardOperationCached && nonCausal {
		program.Operation = ForwardOperationBidirectional
	}
	return program
}
