package inference

const cohere2MTPStateMagic = "L2GC2M01"

// SaveCohere2MTPSession: model-bound resumable state.
func (r *Runner) SaveCohere2MTPSession(session *Cohere2MTPSession) ([]byte, error) {
	return r.saveSingleHeadMTPSession(session, singleHeadMTPStateCodec{
		magic: cohere2MTPStateMagic, label: "Cohere2-MoE MTP",
		validateModel: r.validateCohere2MTP, validateSession: r.validateCohere2MTPSession,
	})
}

// LoadCohere2MTPSession: bounded model-bound restore.
func (r *Runner) LoadCohere2MTPSession(data []byte) (*Cohere2MTPSession, error) {
	return r.loadSingleHeadMTPSession(data, singleHeadMTPStateCodec{
		magic: cohere2MTPStateMagic, label: "Cohere2-MoE MTP",
		validateModel: r.validateCohere2MTP, validateSession: r.validateCohere2MTPSession,
	})
}
