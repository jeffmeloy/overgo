package tokenizer

import (
	"errors"
	"fmt"
)

const (
	MetadataDNAK             = "overgo.tokenizer.dna.k"
	MetadataDNAStartID       = "overgo.tokenizer.dna.start_id"
	MetadataDNAVocabulary    = "overgo.tokenizer.dna.vocabulary"
	MetadataDNASpecialTokens = "overgo.tokenizer.dna.special_tokens"
	MetadataDNAAutoTags      = "overgo.tokenizer.dna.auto_tags"
)

const dnaAlphabet = "ATCG"

type dnaExtension struct {
	k             uint32
	start         uint32
	vocabulary    uint32
	specialTokens []string
	autoTags      bool
}

func ValidateDNAExtension(k, start, vocabulary uint32, specialTokens []string, modelVocabulary uint32) error {
	if k == 0 || k > 15 {
		return fmt.Errorf("k=%d is outside 1..15", k)
	}
	if start == 0 || vocabulary == 0 || start > modelVocabulary || vocabulary > modelVocabulary-start {
		return fmt.Errorf("range [%d,%d) exceeds model vocabulary %d", start, start+vocabulary, modelVocabulary)
	}
	if len(specialTokens) == 0 {
		return errors.New("special token list is empty")
	}
	needed := uint64(len(specialTokens)) + dnaKmerCount(k)
	if needed > uint64(vocabulary) {
		return fmt.Errorf("vocabulary %d is smaller than %d declared tokens", vocabulary, needed)
	}
	for index, token := range specialTokens {
		if token == "" {
			return fmt.Errorf("special token %d is empty", index)
		}
	}
	return nil
}

func dnaKmerCount(k uint32) uint64 { return uint64(1) << (2 * k) }

func (d *dnaExtension) piece(id TokenID) (string, bool) {
	if d == nil || id < TokenID(d.start) {
		return "", false
	}
	offset := uint64(uint32(id) - d.start)
	if offset < uint64(len(d.specialTokens)) {
		return d.specialTokens[offset], true
	}
	index := offset - uint64(len(d.specialTokens))
	if index >= dnaKmerCount(d.k) {
		return "", false
	}
	piece := make([]byte, d.k)
	for cursor := len(piece) - 1; cursor >= 0; cursor-- {
		piece[cursor] = dnaAlphabet[index&3]
		index >>= 2
	}
	return string(piece), true
}
