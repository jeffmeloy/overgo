package runrecord

// Normalize wrappers admit documents in any field order and return canonical
// bytes; each codec owns its canonical-form fact (see artifact.DocumentCodec).

func NormalizeRun(data []byte) (Run, []byte, error) {
	return runCodec.Normalize(data)
}

func NormalizeEvaluation(data []byte) (Evaluation, []byte, error) {
	return evaluationCodec.Normalize(data)
}

func NormalizeEnvironment(data []byte) (Environment, []byte, error) {
	return environmentCodec.Normalize(data)
}

func NormalizeAdvisory(data []byte) (Advisory, []byte, error) {
	return advisoryCodec.Normalize(data)
}

func NormalizeGateResult(data []byte) (GateResult, []byte, error) {
	return gateCodec.Normalize(data)
}
