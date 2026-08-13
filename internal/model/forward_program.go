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
	Operation             ForwardOperation
	Session               ForwardSession
	continuousBatch       bool
	persistentDeviceCache bool
	layerCapture          bool
}

func compileForwardProgram(profile ArchitectureProfile, nonCausal bool) ForwardProgram {
	forward := resolveForwardProgram(profile.Forward, nonCausal)
	cached := forward.Operation == ForwardOperationCached
	altUp := profile.Has(ArchitectureAltUp)
	forward.continuousBatch = cached
	forward.persistentDeviceCache = cached && !altUp && !profile.Has(ArchitectureLatent)
	forward.layerCapture = cached && !altUp
	return forward
}

func (p ForwardProgram) ContinuousBatch() bool       { return p.continuousBatch }
func (p ForwardProgram) PersistentDeviceCache() bool { return p.persistentDeviceCache }
func (p ForwardProgram) LayerCapture() bool          { return p.layerCapture }

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
