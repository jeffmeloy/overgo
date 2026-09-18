package longform

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/tokenizer"
)

// Protocol identifies generation semantics independently of the measured source.
type Protocol string

const (
	// RawContinuation preserves the model's natural end-of-generation choice.
	RawContinuation Protocol = "raw-continuation/v1"
	// GuardContinuation retains sampled EOG tokens and continues to the full
	// declared budget without changing greedy selection or its device path.
	// It is not a natural-stop behavior claim.
	GuardContinuation Protocol = "guard-continuation/fixed-budget/v1"
)

func (protocol Protocol) validate() error {
	if protocol != RawContinuation && protocol != GuardContinuation {
		return fmt.Errorf("longform: unknown generation protocol %q", protocol)
	}
	return nil
}

// Inputs binds a comparison to its checkpoint, corpus, tokenization and protocol.
type Inputs struct {
	Model        artifact.ID `json:"model"`
	CorpusDigest string      `json:"corpus_digest"`
	TokenDigest  string      `json:"token_digest"`
	Protocol     Protocol    `json:"protocol"`
}

// BindInputs identifies the complete fixed corpus, independently of ladder height.
func BindInputs(model artifact.ID, corpus string, tokens []tokenizer.TokenID, protocol Protocol) Inputs {
	text := sha256.Sum256([]byte(corpus))
	hash := sha256.New()
	encoded := make([]byte, binary.Size(tokenizer.TokenID(0)))
	for _, token := range tokens {
		binary.LittleEndian.PutUint32(encoded, uint32(token))
		hash.Write(encoded)
	}
	return Inputs{Model: model, CorpusDigest: hex.EncodeToString(text[:]), TokenDigest: hex.EncodeToString(hash.Sum(nil)), Protocol: protocol}
}

func (inputs Inputs) valid() bool {
	if inputs.Model.Kind() != artifact.KindModel || !inputs.Model.Valid() || inputs.Protocol.validate() != nil {
		return false
	}
	for _, digest := range []string{inputs.CorpusDigest, inputs.TokenDigest} {
		if !artifact.ValidHexDigest(digest) {
			return false
		}
	}
	return true
}
