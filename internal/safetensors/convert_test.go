package safetensors

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"testing"
)

func TestF32ReaderPromotes16BitStorage(t *testing.T) {
	for _, test := range []struct {
		dataType string
		encoded  []byte
		want     []float32
	}{
		{"BF16", []byte{0x80, 0x3f, 0x00, 0xc0}, []float32{1, -2}},
		{"F16", []byte{0x00, 0x3c, 0x00, 0xc0}, []float32{1, -2}},
	} {
		t.Run(test.dataType, func(t *testing.T) {
			tensor, err := NewTensor("x", test.dataType, []uint64{2}, bytes.NewReader(test.encoded), 0, 4)
			if err != nil {
				t.Fatal(err)
			}
			reader, err := F32Reader(tensor)
			if err != nil {
				t.Fatal(err)
			}
			actual, err := io.ReadAll(reader)
			if err != nil {
				t.Fatal(err)
			}
			for index, want := range test.want {
				got := math.Float32frombits(binary.LittleEndian.Uint32(actual[index*4:]))
				if got != want {
					t.Fatalf("value %d = %g, want %g", index, got, want)
				}
			}
		})
	}
}

func TestReadF32PromotesIntoFinalSlab(t *testing.T) {
	for _, test := range []struct {
		dataType string
		encoded  []byte
	}{
		{"F32", []byte{0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0}},
		{"BF16", []byte{0x80, 0x3f, 0x00, 0xc0}},
		{"F16", []byte{0x00, 0x3c, 0x00, 0xc0}},
	} {
		t.Run(test.dataType, func(t *testing.T) {
			tensor, err := NewTensor("x", test.dataType, []uint64{2}, bytes.NewReader(test.encoded), 0, int64(len(test.encoded)))
			if err != nil {
				t.Fatal(err)
			}
			got, err := ReadF32(tensor)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 2 || got[0] != 1 || got[1] != -2 {
				t.Fatalf("values = %v", got)
			}
		})
	}
}

func TestSourceReadAllF32UsesTypedTensorConversion(t *testing.T) {
	f32 := []byte{0x00, 0x00, 0x80, 0x3f}
	bf16 := []byte{0x00, 0xc0}
	first, err := NewTensor("first", "F32", []uint64{1}, bytes.NewReader(f32), 0, int64(len(f32)))
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewTensor("second", "BF16", []uint64{1}, bytes.NewReader(bf16), 0, int64(len(bf16)))
	if err != nil {
		t.Fatal(err)
	}
	values, err := (&Source{Tensors: map[string]Tensor{"first": first, "second": second}}).ReadAllF32()
	if err != nil {
		t.Fatal(err)
	}
	if values["first"][0] != 1 || values["second"][0] != -2 {
		t.Fatalf("values = %v", values)
	}
}

func TestReadBF16RetainsNativeWords(t *testing.T) {
	encoded := []byte{0x80, 0x3f, 0x00, 0xc0}
	tensor, err := NewTensor("x", "BF16", []uint64{2}, bytes.NewReader(encoded), 0, int64(len(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	got, err := ReadBF16(tensor)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != 0x3f80 || got[1] != 0xc000 {
		t.Fatalf("words = %x", got)
	}
}
