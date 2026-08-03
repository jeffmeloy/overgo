package inference

const qwen35MTPStateMagic = "L2GMTP01"

// SaveQwen35MTPSession: draft/target-bound resumable state.
func (r *Runner) SaveQwen35MTPSession(session *Qwen35MTPSession) ([]byte, error) {
	return r.saveSingleHeadMTPSession(session, singleHeadMTPStateCodec{
		magic: qwen35MTPStateMagic, label: "Qwen3.5 MTP",
		validateModel: r.validateQwen35MTP, validateSession: r.validateQwen35MTPSession,
	})
}

// LoadQwen35MTPSession: bounded model-bound restore.
func (r *Runner) LoadQwen35MTPSession(data []byte) (*Qwen35MTPSession, error) {
	return r.loadSingleHeadMTPSession(data, singleHeadMTPStateCodec{
		magic: qwen35MTPStateMagic, label: "Qwen3.5 MTP",
		validateModel: r.validateQwen35MTP, validateSession: r.validateQwen35MTPSession,
	})
}
