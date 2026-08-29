package gemma4convert

import (
	"encoding/binary"
	"io"
	"math"
	"os"
	"slices"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
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
	weight, err := safetensors.NewTensor("weight", "F8_E4M3", []uint64{2, 4}, file, 0, int64(len(weights)))
	if err != nil {
		t.Fatal(err)
	}
	scale, err := safetensors.NewTensor("scale", "F32", []uint64{2}, file, int64(len(weights)), int64(len(encodedScales)))
	if err != nil {
		t.Fatal(err)
	}
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
		want := dtype.Float32ToBF16(dtype.F8E4M3ToFloat32(encoded) * scales[index/4])
		got := binary.LittleEndian.Uint16(actual[index*2:])
		if got != want {
			t.Fatalf("value %d = %#04x, want %#04x", index, got, want)
		}
	}
}

// TestFP8NativeReaderConcatenatesWeightThenScale: the fp8-preserving emit path
// writes the exact resident buffer layout -- all e4m3 weight bytes followed by
// the per-row F32 scale bytes -- as a raw concatenation with no decode/repack.
func TestFP8NativeReaderConcatenatesWeightThenScale(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "fp8native-*.bin")
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
	weight, err := safetensors.NewTensor("weight", "F8_E4M3", []uint64{2, 4}, file, 0, int64(len(weights)))
	if err != nil {
		t.Fatal(err)
	}
	scale, err := safetensors.NewTensor("scale", "F32", []uint64{2}, file, int64(len(weights)), int64(len(encodedScales)))
	if err != nil {
		t.Fatal(err)
	}
	reader, err := newFP8NativeReader(weight, scale)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	want := append(slices.Clone(weights), encodedScales...)
	if len(actual) != len(want) {
		t.Fatalf("output bytes = %d, want %d", len(actual), len(want))
	}
	for index := range want {
		if actual[index] != want[index] {
			t.Fatalf("byte %d = %#02x, want %#02x", index, actual[index], want[index])
		}
	}
}

func TestPositionReaderTransposesPositionAndAxis(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "position-*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	for value := range uint16(8) {
		var encoded [2]byte
		binary.LittleEndian.PutUint16(encoded[:], value)
		if _, err := file.Write(encoded[:]); err != nil {
			t.Fatal(err)
		}
	}
	source, err := safetensors.NewTensor("position", "BF16", []uint64{2, 2, 2}, file, 0, 16)
	if err != nil {
		t.Fatal(err)
	}
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
	for value := range uint16(12) {
		var encoded [2]byte
		binary.LittleEndian.PutUint16(encoded[:], value)
		if _, err := file.Write(encoded[:]); err != nil {
			t.Fatal(err)
		}
	}
	source, err := safetensors.NewTensor("patch", "BF16", []uint64{2, 6}, file, 0, 24)
	if err != nil {
		t.Fatal(err)
	}
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
