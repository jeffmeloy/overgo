package gguf

import (
	"bytes"
	"encoding/binary"
	"io"
	"testing"

	"overgo/internal/testcheck"
)

// TestAViewTensorIsReleasedAtItsEnd opens a view whose first built tensor is larger than the directory's read-ahead,
// which builds it and reads a prefix of it, then reads it through in four parts as a load does: it is built once, by
// the open, kept through the load's reads, and released by the read that reaches its last byte.
func TestAViewTensorIsReleasedAtItsEnd(t *testing.T) {
	t.Parallel()
	const elements = 2 * directoryReadBufferBytes / 4
	data := make([]byte, 4*elements)
	for index := range elements {
		binary.LittleEndian.PutUint32(data[4*index:], uint32(index))
	}
	built := 0
	file, err := View(nil, []ViewTensor{{Name: "large", Shape: []uint64{elements}, Type: DTypeF32, Open: func() (io.Reader, error) {
		built++
		return bytes.NewReader(data), nil
	}}}, WriteOptions{}, "")
	testcheck.NoError(t, err)
	source := file.source.(*viewSource)
	info, _ := file.Tensor("large")
	part, read := info.Size/4, make([]byte, 0, info.Size)
	for from := uint64(0); from < info.Size; from += part {
		chunk := make([]byte, part)
		testcheck.NoError(t, file.ReadTensorRange(info, from, chunk))
		if read = append(read, chunk...); from+part < info.Size && len(source.built) != 1 {
			t.Fatalf("after %d of %d bytes the view keeps %d built tensors, want the one being read", from+part, info.Size, len(source.built))
		}
	}
	if !bytes.Equal(read, data) || built != 1 || len(source.built) != 0 {
		t.Fatalf("the load read equal=%v, the tensor was built %d times and %d are kept, want it built once and released at its end",
			bytes.Equal(read, data), built, len(source.built))
	}
}
