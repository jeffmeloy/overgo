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
