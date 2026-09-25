package sampling

import (
	"crypto/sha256"
	"encoding/binary"
	"strconv"
)

// The cache stores only SHA-256 keys and signatures, never copies of grammar
// sources or vocabulary bytes. One runner's vocabulary retains at most this
// many warmed declarations, with deterministic FIFO replacement.
const gbnfSignatureCacheCapacity = 32

func (v *GBNFVocabulary) cachedGrammarSignature(source, root string, lazy GBNFLazyOptions, grammar *GBNFGrammar) uint64 {
	key := gbnfSignatureCacheKey(source, root, lazy)
	v.signatureMu.Lock()
	signature, found := v.signatures[key]
	v.signatureMu.Unlock()
	if found {
		return signature
	}
	// Never hold the binding lock through the vocabulary scan: distinct
	// grammars may compile concurrently. Duplicate cold scans remain valid.
	signature = grammarSignature(source, root, v.pieces, v.eos, grammar)
	v.signatureMu.Lock()
	defer v.signatureMu.Unlock()
	if prior, loaded := v.signatures[key]; loaded {
		return prior
	}
	if v.signatures == nil {
		v.signatures = make(map[[32]byte]uint64, gbnfSignatureCacheCapacity)
	}
	if v.signatureUsed == gbnfSignatureCacheCapacity {
		delete(v.signatures, v.signatureOrder[v.signatureNext])
	} else {
		v.signatureUsed++
	}
	v.signatureOrder[v.signatureNext] = key
	v.signatureNext = (v.signatureNext + 1) % gbnfSignatureCacheCapacity
	v.signatures[key] = signature
	return signature
}

func gbnfSignatureCacheKey(source, root string, lazy GBNFLazyOptions) [32]byte {
	hash := sha256.New()
	var length [8]byte
	writeBytes := func(value []byte) {
		binary.LittleEndian.PutUint64(length[:], uint64(len(value)))
		_, _ = hash.Write(length[:])
		_, _ = hash.Write(value)
	}
	writeBytes([]byte(source))
	writeBytes([]byte(root))
	writeBytes([]byte(strconv.FormatBool(lazy.Enabled)))
	binary.LittleEndian.PutUint64(length[:], uint64(len(lazy.Tokens)))
	_, _ = hash.Write(length[:])
	for _, token := range lazy.Tokens {
		binary.LittleEndian.PutUint64(length[:], uint64(token))
		_, _ = hash.Write(length[:])
	}
	binary.LittleEndian.PutUint64(length[:], uint64(len(lazy.Patterns)))
	_, _ = hash.Write(length[:])
	for _, pattern := range lazy.Patterns {
		writeBytes([]byte(pattern))
	}
	var key [32]byte
	copy(key[:], hash.Sum(nil))
	return key
}
