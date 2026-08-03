package statecodec

import (
	"errors"
	"testing"
)

func TestCodecRoundTrip(t *testing.T) {
	encoder := NewEncoder(32)
	encoder.Raw([]byte("MAGIC001"))
	encoder.U32(7)
	encoder.U64(11)
	encoder.F32(1.5)
	data, err := encoder.Data()
	if err != nil {
		t.Fatal(err)
	}
	decoder := NewDecoder(data, 32)
	if string(decoder.Raw(8)) != "MAGIC001" || decoder.U32() != 7 || decoder.U64() != 11 || decoder.F32() != 1.5 {
		t.Fatal("round trip differs")
	}
	if err := decoder.Done(); err != nil {
		t.Fatal(err)
	}
}

func TestCodecBounds(t *testing.T) {
	encoder := NewEncoder(3)
	encoder.U32(1)
	if _, err := encoder.Data(); !errors.Is(err, ErrLimit) {
		t.Fatalf("error = %v", err)
	}
	decoder := NewDecoder([]byte{1}, 1)
	decoder.U32()
	if !errors.Is(decoder.Err(), ErrTruncated) {
		t.Fatalf("error = %v", decoder.Err())
	}
}
