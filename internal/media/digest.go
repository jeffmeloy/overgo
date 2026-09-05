package media

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// SamplesSHA256 hashes the exact little-endian float32 representation of signal
// samples or derived features, without allocating an encoded copy. It does not
// canonicalize signed zero or non-finite values; admission is the caller's job.
func SamplesSHA256(samples []float32) string {
	digest := sha256.New()
	var word [4]byte
	for _, sample := range samples {
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(sample))
		_, _ = digest.Write(word[:])
	}
	return hex.EncodeToString(digest.Sum(nil))
}
