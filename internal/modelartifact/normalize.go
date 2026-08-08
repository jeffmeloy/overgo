package modelartifact

// Normalize wrappers admit documents in any field order and return canonical
// bytes; each codec owns its canonical-form fact (see artifact.DocumentCodec).

func NormalizeTensorInventoryDocument(data []byte) (TensorInventoryDocument, []byte, error) {
	return tensorInventoryCodec.Normalize(data)
}

func NormalizeTensorMeasurementDocument(data []byte) (TensorMeasurementDocument, []byte, error) {
	return tensorMeasurementCodec.Normalize(data)
}
