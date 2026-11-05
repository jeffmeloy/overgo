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

// LoadTokenizer reads the artifact's tokenizer.model into a Unigram.
func LoadTokenizer(path string) (*Unigram, error) {
	pieces, unkID, err := tokenizer.ReadSentencePieceModel(path)
	if err != nil {
		return nil, err
	}
	return tokenizer.NewUnigram(pieces, unkID)
}
