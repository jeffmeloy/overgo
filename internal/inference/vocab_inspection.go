package inference

import "overgo/internal/tokenizer"

// VocabularyLen: number of entries in the loaded vocabulary, or zero when no
// vocabulary is present. Read-only; safe without the generation mutex.
func (r *Runner) VocabularyLen() int {
	if r == nil || r.vocab == nil {
		return 0
	}
	return r.vocab.Len()
}

// VocabularyToken: the vocabulary entry (text, score, type) for id, for
// read-only inspection by the analysis surface. Reports false for an id outside
// the vocabulary. Read-only; safe without the generation mutex.
func (r *Runner) VocabularyToken(id tokenizer.TokenID) (tokenizer.Token, bool) {
	if r == nil || r.vocab == nil {
		return tokenizer.Token{}, false
	}
	return r.vocab.Token(id)
}
