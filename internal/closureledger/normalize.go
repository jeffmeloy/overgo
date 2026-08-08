package closureledger

// Normalize admits a document in any field order and returns its canonical
// form; the codec owns the canonical-bytes fact (see artifact.DocumentCodec).
func Normalize(data []byte) (Document, []byte, error) {
	return documentCodec.Normalize(data)
}
