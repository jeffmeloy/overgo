package gguf

import (
	"bytes"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestXXH64VectorsAndIncrementalWrites(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"", "ef46db3751d8e999"},
		{"a", "d24ec4f1a98c6e5b"},
		{"hello", "26c7827d889f6da3"},
	}
	for _, test := range tests {
		digest := newXXH64()
		for _, value := range []byte(test.input) {
			digest.Write([]byte{value})
		}
		if got := digest.Sum64(); got != mustParseHex64(t, test.want) {
			t.Fatalf("XXH64(%q) = %016x, want %s", test.input, got, test.want)
		}
	}
	data := make([]byte, 257)
	for index := range data {
		data[index] = byte(index*31 + 7)
	}
	whole := newXXH64()
	whole.Write(data)
	chunked := newXXH64()
	for offset := 0; offset < len(data); offset += 7 {
		end := min(offset+7, len(data))
		chunked.Write(data[offset:end])
	}
	if whole.Sum64() != chunked.Sum64() {
		t.Fatalf("whole/chunked hashes = %016x/%016x", whole.Sum64(), chunked.Sum64())
	}
}

func TestHashMatchesConcatenatedTensorPayloads(t *testing.T) {
	payloads := [][]byte{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	}
	var encoded bytes.Buffer
	if err := Write(
		&encoded,
		nil,
		[]TensorData{
			{Name: "first", Shape: []uint64{1}, Type: DTypeF32, Data: bytes.NewReader(payloads[0])},
			{Name: "second", Shape: []uint64{1}, Type: DTypeF32, Data: bytes.NewReader(payloads[1])},
		},
		WriteOptions{},
	); err != nil {
		t.Fatal(err)
	}
	file, err := Parse(
		bytes.NewReader(encoded.Bytes()),
		uint64(encoded.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := file.Hash(HashOptions{
		XXH64:     true,
		SHA1:      true,
		SHA256:    true,
		UUID:      true,
		PerTensor: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	combined := append(append([]byte(nil), payloads[0]...), payloads[1]...)
	wantSHA1 := sha1.Sum(combined)
	wantSHA256 := sha256.Sum256(combined)
	if result.Model.SHA1 != hex.EncodeToString(wantSHA1[:]) ||
		result.Model.SHA256 != hex.EncodeToString(wantSHA256[:]) {
		t.Fatalf("model hashes = %#v", result.Model)
	}
	if len(result.Tensors) != 2 || result.Tensors[1].Name != "second" {
		t.Fatalf("tensor hashes = %#v", result.Tensors)
	}
	secondSHA256 := sha256.Sum256(payloads[1])
	if result.Tensors[1].Values.SHA256 != hex.EncodeToString(secondSHA256[:]) {
		t.Fatalf("second SHA-256 = %s", result.Tensors[1].Values.SHA256)
	}
	if len(result.Model.UUID) != 36 {
		t.Fatalf("UUID = %q", result.Model.UUID)
	}
}

func mustParseHex64(t *testing.T, value string) uint64 {
	t.Helper()
	var result uint64
	for _, digit := range []byte(value) {
		result <<= 4
		switch {
		case digit >= '0' && digit <= '9':
			result |= uint64(digit - '0')
		case digit >= 'a' && digit <= 'f':
			result |= uint64(digit-'a') + 10
		default:
			t.Fatalf("invalid hex digit %q", digit)
		}
	}
	return result
}
