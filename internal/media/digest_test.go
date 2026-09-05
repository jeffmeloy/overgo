package media

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"testing"
)

func TestSamplesSHA256ExactRepresentation(t *testing.T) {
	// IEEE 754 little-endian encodings of 1, -2, and negative zero.
	encoded := []byte{0, 0, 128, 63, 0, 0, 0, 192, 0, 0, 0, 128}
	want := sha256.Sum256(encoded)
	negativeZero := float32(math.Copysign(0, -1))
	if got := SamplesSHA256([]float32{1, -2, negativeZero}); got != hex.EncodeToString(want[:]) {
		t.Fatalf("sample digest = %s, want %x", got, want)
	}
	if SamplesSHA256([]float32{0}) == SamplesSHA256([]float32{negativeZero}) {
		t.Fatal("signed zero must retain its distinct bit identity")
	}
	if SamplesSHA256(nil) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
		t.Fatal("empty digest differs from SHA256 empty message")
	}
}
