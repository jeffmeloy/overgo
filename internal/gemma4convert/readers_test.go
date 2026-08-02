package gemma4convert

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"testing"
)

func TestFP8BF16ReaderFoldsRowScales(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "fp8-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	weights := []byte{0x00, 0x01, 0x08, 0x38, 0x40, 0x7e, 0x80, 0xb8}
	scales := []float32{2, 0.5}
	encodedScales := make([]byte, len(scales)*4)
	for index, scale := range scales {
		binary.LittleEndian.PutUint32(encodedScales[index*4:], math.Float32bits(scale))
	}
	if _, err := file.Write(append(weights, encodedScales...)); err != nil {
		t.Fatal(err)
	}
	weight := Tensor{Name: "weight", DType: "F8_E4M3", Shape: []uint64{2, 4}, file: file, size: int64(len(weights))}
	scale := Tensor{Name: "scale", DType: "F32", Shape: []uint64{2}, file: file, offset: int64(len(weights)), size: int64(len(encodedScales))}
	reader, err := newFP8BF16Reader(weight, scale)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	if len(actual) != len(weights)*2 {
		t.Fatalf("output bytes = %d, want %d", len(actual), len(weights)*2)
	}
	for index, encoded := range weights {
		want := float32ToBF16(fp8E4M3FN(encoded) * scales[index/4])
		got := binary.LittleEndian.Uint16(actual[index*2:])
		if got != want {
			t.Fatalf("value %d = %#04x, want %#04x", index, got, want)
		}
	}
}

func TestPositionReaderTransposesPositionAndAxis(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "position-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	for value := uint16(0); value < 8; value++ {
		var encoded [2]byte
		binary.LittleEndian.PutUint16(encoded[:], value)
		if _, err := file.Write(encoded[:]); err != nil {
			t.Fatal(err)
		}
	}
	source := Tensor{DType: "BF16", Shape: []uint64{2, 2, 2}, file: file, size: 16}
	reader, err := newPositionReader(source)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint16{0, 1, 4, 5, 2, 3, 6, 7}
	for index, expected := range want {
		got := binary.LittleEndian.Uint16(actual[index*2:])
		if got != expected {
			t.Fatalf("value %d = %d, want %d", index, got, expected)
		}
	}
}

func TestPatchPermutationReaderConvertsInterleavedToPlanar(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "patch-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	for value := uint16(0); value < 12; value++ {
		var encoded [2]byte
		binary.LittleEndian.PutUint16(encoded[:], value)
		if _, err := file.Write(encoded[:]); err != nil {
			t.Fatal(err)
		}
	}
	source := Tensor{DType: "BF16", Shape: []uint64{2, 6}, file: file, size: 24}
	reader, err := newPatchPermutationReader(source)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint16{0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11}
	for index, expected := range want {
		got := binary.LittleEndian.Uint16(actual[index*2:])
		if got != expected {
			t.Fatalf("value %d = %d, want %d", index, got, expected)
		}
	}
}

func TestFP8E4M3FNValues(t *testing.T) {
	for _, test := range []struct {
		encoded byte
		want    float32
	}{
		{0x01, 1.0 / 512.0},
		{0x08, 1.0 / 64.0},
		{0x38, 1},
		{0x7e, 448},
		{0xb8, -1},
	} {
		if got := fp8E4M3FN(test.encoded); got != test.want {
			t.Fatalf("decode %#02x = %g, want %g", test.encoded, got, test.want)
		}
	}
	if !math.IsNaN(float64(fp8E4M3FN(0x7f))) {
		t.Fatal("0x7f is not NaN")
	}
}
