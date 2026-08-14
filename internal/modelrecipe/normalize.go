package modelrecipe

// Normalize wrappers admit documents in any field order and return canonical
// bytes; each codec owns its canonical-form fact (see artifact.DocumentCodec).

func NormalizeProfileDocument(data []byte) (ProfileDocument, []byte, error) {
	return profileCodec.Normalize(data)
}

func NormalizeCatalogProfileDerivation(data []byte) (CatalogProfileDerivation, []byte, error) {
	return catalogProfileDerivationCodec.Normalize(data)
}

func NormalizeModelDefinitionDocument(data []byte) (ModelDefinitionDocument, []byte, error) {
	return modelDefinitionCodec.Normalize(data)
}
