package longform

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"

	"overgo/internal/artifact"
	"overgo/internal/tokenizer"
)

const rawContinuationProtocol = "raw-continuation/v1"

// Inputs binds a comparison to its checkpoint, corpus, tokenization and protocol.
type Inputs struct {
	Model        artifact.ID `json:"model"`
	CorpusDigest string      `json:"corpus_digest"`
	TokenDigest  string      `json:"token_digest"`
	Protocol     string      `json:"protocol"`
}

// BindInputs identifies the complete fixed corpus, independently of ladder height.
func BindInputs(model artifact.ID, corpus string, tokens []tokenizer.TokenID) Inputs {
	text := sha256.Sum256([]byte(corpus))
	hash := sha256.New()
	encoded := make([]byte, binary.Size(tokenizer.TokenID(0)))
	for _, token := range tokens {
		binary.LittleEndian.PutUint32(encoded, uint32(token))
		hash.Write(encoded)
	}
	return Inputs{Model: model, CorpusDigest: hex.EncodeToString(text[:]), TokenDigest: hex.EncodeToString(hash.Sum(nil)), Protocol: rawContinuationProtocol}
}

func (inputs Inputs) valid() bool {
	if inputs.Model.Kind() != artifact.KindModel || !inputs.Model.Valid() || inputs.Protocol != rawContinuationProtocol {
		return false
	}
	for _, digest := range []string{inputs.CorpusDigest, inputs.TokenDigest} {
		decoded, err := hex.DecodeString(digest)
		if err != nil || len(decoded) != sha256.Size {
			return false
		}
	}
	return true
}
