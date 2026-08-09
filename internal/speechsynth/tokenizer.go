// Text-conditioner tokenizer entry points. The unigram Viterbi table and the
// SentencePiece ModelProto reader were promoted to internal/tokenizer (one
// owner, shared with latent-video text conditioning); these names remain for
// package callers.
package speechsynth

import "overgo/internal/tokenizer"

// Piece is one unigram vocabulary entry.
type Piece = tokenizer.UnigramPiece

// Unigram owns Viterbi policy over a parsed piece table.
type Unigram = tokenizer.Unigram

// NewUnigram builds the table; unkID must index a real piece.
func NewUnigram(pieces []Piece, unkID int) (*Unigram, error) {
	return tokenizer.NewUnigram(pieces, unkID)
}

// ReadModelFile parses a vendor tokenizer.model piece table.
func ReadModelFile(path string) ([]Piece, int, error) {
	return tokenizer.ReadSentencePieceModel(path)
}

// LoadTokenizer reads the artifact's tokenizer.model into a Unigram.
func LoadTokenizer(path string) (*Unigram, error) {
	pieces, unkID, err := ReadModelFile(path)
	if err != nil {
		return nil, err
	}
	return NewUnigram(pieces, unkID)
}
