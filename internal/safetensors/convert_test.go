package safetensors

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
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
			tensor := testcheck.Must(NewTensor("x", test.dataType, []uint64{2}, bytes.NewReader(test.encoded), 0, 4))(t)
			reader := testcheck.Must(F32Reader(tensor))(t)
			actual := testcheck.Must(io.ReadAll(reader))(t)
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
			tensor := testcheck.Must(NewTensor("x", test.dataType, []uint64{2}, bytes.NewReader(test.encoded), 0, int64(len(test.encoded))))(t)
			got := testcheck.Must(ReadF32(tensor))(t)
			if len(got) != 2 || got[0] != 1 || got[1] != -2 {
				t.Fatalf("values = %v", got)
			}
		})
	}
}

func TestSourceMaterializeF32SelectsValuesAndShapes(t *testing.T) {
	ignored := make([]byte, binary.Size(uint64(0)))
	bf16 := []byte{0x00, 0xc0}
	first := testcheck.Must(NewTensor("first", "I64", []uint64{1}, bytes.NewReader(ignored), 0, int64(len(ignored))))(t)
	second := testcheck.Must(NewTensor("second", "BF16", []uint64{1}, bytes.NewReader(bf16), 0, int64(len(bf16))))(t)
	catalog := testcheck.Must((&Source{Tensors: map[string]Tensor{"first": first, "second": second}}).MaterializeF32(F32Selection{
		Keep:         func(name string) bool { return name == "second" },
		RetainShapes: true,
	}))(t)
	if len(catalog.Values) != 1 || catalog.Values["second"][0] != -2 {
		t.Fatalf("values = %v", catalog.Values)
	}
	if len(catalog.Shapes) != 1 || !slices.Equal(catalog.Shapes["second"], []int{1}) {
		t.Fatalf("shapes = %v", catalog.Shapes)
	}
}

func TestReadBF16RetainsNativeWords(t *testing.T) {
	encoded := []byte{0x80, 0x3f, 0x00, 0xc0}
	tensor := testcheck.Must(NewTensor("x", "BF16", []uint64{2}, bytes.NewReader(encoded), 0, int64(len(encoded))))(t)
	got := testcheck.Must(ReadBF16(tensor))(t)
	if len(got) != 2 || got[0] != 0x3f80 || got[1] != 0xc000 {
		t.Fatalf("words = %x", got)
	}
}

func TestReadF32RejectsUnsupportedBeforeAllocation(t *testing.T) {
	// A valid U8 extent exceeds the maximum allocatable F32 slice. Reject its
	// unsupported conversion before attempting allocation or touching payload.
	tensor := testcheck.Must(NewTensor("unsupported", "U8", []uint64{math.MaxInt64}, bytes.NewReader(nil), 0, math.MaxInt64))(t)
	values, err := ReadF32(tensor)
	if values != nil || err == nil || !strings.Contains(err.Error(), "cannot promote U8") {
		t.Fatalf("unsupported conversion returned values=%v error=%v", values, err)
	}
}

func TestReadF32RejectsTruncatedPayload(t *testing.T) {
	for _, storage := range []string{"F32", "F16", "BF16"} {
		t.Run(storage, func(t *testing.T) {
			shape := []uint64{1}
			size := testcheck.Must(TensorBytes(storage, shape))(t)
			tensor := testcheck.Must(NewTensor("truncated", storage, shape, bytes.NewReader([]byte{0}), 0, int64(size)))(t)
			values, err := ReadF32(tensor)
			if err == nil || values != nil {
				t.Fatalf("truncated conversion returned values=%v error=%v", values, err)
			}
		})
	}
}
