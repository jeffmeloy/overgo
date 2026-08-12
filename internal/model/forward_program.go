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

func compileForwardProgram(profile ArchitectureProfile) ForwardProgram {
	switch profile.Forward {
	case ForwardNonCausal:
		return ForwardProgram{Operation: ForwardOperationBidirectional}
	case ForwardWavTokenizer:
		return ForwardProgram{Operation: ForwardOperationAudioTokens}
	case ForwardT5Encoder:
		return ForwardProgram{Operation: ForwardOperationEncoder}
	case ForwardDFlash:
		return ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionPairedFeatures}
	case ForwardEagle3:
		return ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionFeatureDraft}
	case ForwardGemma4Assistant:
		return ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionPairedProjection}
	case ForwardT5:
		return ForwardProgram{Operation: ForwardOperationSession, Session: ForwardSessionEncoderDecoder}
	default:
		return ForwardProgram{Operation: ForwardOperationCached}
	}
}

func (p ForwardProgram) valid() bool {
	if p.Operation >= forwardOperationCount || p.Session >= forwardSessionCount {
		return false
	}
	return (p.Operation == ForwardOperationSession) == (p.Session != ForwardSessionNone)
}
