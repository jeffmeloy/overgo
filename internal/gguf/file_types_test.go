package gguf

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestGGUFRefusesTheInt8ConvRotType holds the GGUF boundary to the types a GGUF file carries (gguf-int8-boundary,
// SQA-419): writing a 256-by-4 tensor of dtype.I8ConvRot, the CUDA int8 ConvRot product's own type, fails before any
// byte is written, and a file whose tensor names type 44 is refused as it is read, while the same tensor in F8E4M3, the
// Gemma converter's extension, writes and reads back. Since moss-int8/product the writer had emitted type 44 and the
// reader had read it as i8_convrot.
func TestGGUFRefusesTheInt8ConvRotType(t *testing.T) {
	t.Parallel()
	const name, columns, rows = "weight", 256, 4
	tensor := func(storage DType) []TensorData {
		return []TensorData{{Name: name, Shape: []uint64{columns, rows}, Type: storage, Data: bytes.NewReader(make([]byte, columns*rows+rows*4))}}
	}
	var refused bytes.Buffer
	if err := Write(&refused, nil, tensor(dtype.I8ConvRot), WriteOptions{}); err == nil || refused.Len() != 0 {
		t.Fatalf("writing an int8 ConvRot tensor wrote %d bytes and returned %v; want it refused before any byte", refused.Len(), err)
	}
	var written bytes.Buffer
	testcheck.NoError(t, Write(&written, nil, tensor(DTypeF8E4M3), WriteOptions{}))
	file := testcheck.Must(Parse(bytes.NewReader(written.Bytes()), uint64(written.Len()), DefaultOptions()))(t)
	if len(file.Tensors) != 1 || file.Tensors[0].Type != DTypeF8E4M3 {
		t.Fatalf("an F8E4M3 tensor read back as %+v; want its one tensor in F8E4M3", file.Tensors)
	}
	// A tensor's record is its name's length and bytes, its dimension count, its dimensions and then its type.
	record := binary.LittleEndian.AppendUint64(nil, uint64(len(name)))
	at := bytes.Index(written.Bytes(), append(record, name...))
	if at < 0 {
		t.Fatal("the written file holds no record of its tensor")
	}
	typeAt := at + len(record) + len(name) + 4 + 2*8
	patched := bytes.Clone(written.Bytes())
	binary.LittleEndian.PutUint32(patched[typeAt:], uint32(dtype.I8ConvRot))
	if _, err := Parse(bytes.NewReader(patched), uint64(len(patched)), DefaultOptions()); err == nil || !strings.Contains(err.Error(), "no GGUF file carries") {
		t.Fatalf("a file whose tensor names type %d read with %v; want it refused as no GGUF file type", dtype.I8ConvRot, err)
	}
}
