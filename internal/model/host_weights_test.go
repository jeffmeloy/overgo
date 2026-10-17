package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestLoadHostTensor(t *testing.T) {
	data := hostTensorFixture(t, "weight", 4)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	value, err := LoadHostTensor(t.Context(), file, file.Tensors[0])
	if err != nil {
		t.Fatal(err)
	}
	if !value.Shape.Equal(tensor.MustShape(4)) {
		t.Fatalf("shape = %v", value.Shape.Slice())
	}
	for index, want := range []float32{1, 2, 3, 4} {
		if value.Data[index] != want {
			t.Fatalf("value[%d] = %v, want %v", index, value.Data[index], want)
		}
	}
}

func TestLoadHostRowsAndDotRows(t *testing.T) {
	data := hostTensorFixture(t, "table", 2, 3)
	reader := &countedRowReader{Reader: bytes.NewReader(data)}
	file, err := gguf.Parse(reader, uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	reader.calls = 0
	value, err := LoadHostRows(t.Context(), file, file.Tensors[0], []uint32{2, 0, 2})
	if err != nil {
		t.Fatal(err)
	}
	if reader.calls != 2 {
		t.Fatalf("row read requests = %d, want 2", reader.calls)
	}
	want := []float32{5, 6, 1, 2, 5, 6}
	for index := range want {
		if value.Data[index] != want[index] {
			t.Fatalf("row value[%d] = %v, want %v", index, value.Data[index], want[index])
		}
	}
	scores, err := DotRows(t.Context(), file, file.Tensors[0], []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	wantScores := []float32{3, 7, 11}
	for row, want := range wantScores {
		if scores[row] != want {
			t.Fatalf("score[%d] = %v, want %v", row, scores[row], want)
		}
	}
	for _, indices := range [][]uint32{{3}, {2, 3}, nil} {
		if _, err := LoadHostRows(t.Context(), file, file.Tensors[0], indices); err == nil {
			t.Fatalf("invalid rows accepted: %v", indices)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	reader.calls, reader.after = 0, cancel
	value, err = LoadHostRows(ctx, file, file.Tensors[0], []uint32{2, 2, 0})
	if !errors.Is(err, context.Canceled) || reader.calls != 1 || len(value.Data) != 0 {
		t.Fatalf("interrupted rows = %v, reads %d, error %v", value.Data, reader.calls, err)
	}
	reader.calls = 0
	if _, err := LoadHostRows(ctx, file, file.Tensors[0], []uint32{0}); !errors.Is(err, context.Canceled) || reader.calls != 0 {
		t.Fatalf("cancelled read touched file %d times: %v", reader.calls, err)
	}
}

type countedRowReader struct {
	*bytes.Reader
	calls int
	after func()
}

func (r *countedRowReader) ReadAt(p []byte, off int64) (int, error) {
	r.calls++
	n, err := r.Reader.ReadAt(p, off)
	if r.after != nil {
		r.after()
	}
	return n, err
}

func hostTensorFixture(t *testing.T, name string, dimensions ...uint64) []byte {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		_, _ = buffer.WriteString(value)
	}
	_, _ = buffer.WriteString(gguf.Magic)
	write(uint32(gguf.CurrentVersion))
	write(uint64(1))
	write(uint64(0))
	writeString(name)
	write(uint32(len(dimensions)))
	elements := uint64(1)
	for _, dimension := range dimensions {
		write(dimension)
		elements *= dimension
	}
	write(uint32(dtype.F32))
	write(uint64(0))
	for buffer.Len()%gguf.DefaultAlignment != 0 {
		_ = buffer.WriteByte(0)
	}
	for index := range elements {
		write(float32(index + 1))
	}
	for buffer.Len()%gguf.DefaultAlignment != 0 {
		_ = buffer.WriteByte(0)
	}
	return buffer.Bytes()
}
