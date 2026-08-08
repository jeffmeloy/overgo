package recipe

// Normalize wrappers admit documents in any field order and return canonical
// bytes; each codec owns its canonical-form fact (see artifact.DocumentCodec).

func NormalizeDefinition(data []byte) (Definition, []byte, error) {
	return definitionCodec.Normalize(data)
}

func NormalizeLifecycleEvent(data []byte) (LifecycleEvent, []byte, error) {
	return lifecycleCodec.Normalize(data)
}

func NormalizeDecision(data []byte) (Decision, []byte, error) {
	return decisionCodec.Normalize(data)
}
