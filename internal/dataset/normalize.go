package dataset

// Normalize wrappers admit documents in any field order and return canonical
// bytes; each codec owns its canonical-form fact (see artifact.DocumentCodec).

func Normalize(data []byte) (Document, []byte, error) {
	return documentCodec.Normalize(data)
}

func NormalizeMembership(data []byte) (Membership, []byte, error) {
	return membershipCodec.Normalize(data)
}
