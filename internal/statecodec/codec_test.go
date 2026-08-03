package statecodec

import (
	"errors"
	"testing"
)

func TestCodecRoundTrip(t *testing.T) {
	encoder := NewEncoderCapacity(64, 48)
	encoder.Raw([]byte("MAGIC001"))
	encoder.U32(7)
	encoder.U64(11)
	encoder.F32(1.5)
	encoder.I32(-9)
	encoder.F64(2.5)
	encoder.String32("state")
	data, err := encoder.Data()
	if err != nil {
		t.Fatal(err)
	}
	decoder := NewDecoder(data, 64)
	if string(decoder.Raw(8)) != "MAGIC001" || decoder.U32() != 7 || decoder.U64() != 11 || decoder.F32() != 1.5 ||
		decoder.I32() != -9 || decoder.F64() != 2.5 || decoder.String32(8) != "state" {
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
	if _, err := NewEncoderCapacity(3, 4).Data(); !errors.Is(err, ErrLimit) {
		t.Fatalf("capacity error = %v", err)
	}
	stringEncoder := NewEncoder(16)
	stringEncoder.String32("state")
	data, err := stringEncoder.Data()
	if err != nil {
		t.Fatal(err)
	}
	stringDecoder := NewDecoder(data, 16)
	if got := stringDecoder.String32(4); got != "" || !errors.Is(stringDecoder.Err(), ErrLimit) {
		t.Fatalf("limited string = %q, error = %v", got, stringDecoder.Err())
	}
}
