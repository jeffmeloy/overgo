package gguf

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// TestRewriteReleasesATensorOnceItIsRead holds a rewritten tensor to one
// computation and to no retained bytes once its last byte is read: the writer
// keeps every tensor's reader until the file is written, so a reader that held
// its encoded bytes would hold the whole rewritten file in memory at the end.
func TestRewriteReleasesATensorOnceItIsRead(t *testing.T) {
	t.Parallel()
	values := quantizeFixtureValues(64)
	var source bytes.Buffer
	err := Write(&source, []Metadata{
		{Key: "general.file_type", Value: Value{Type: ValueTypeUint32, Data: uint32(0)}},
		{Key: "split.no", Value: Value{Type: ValueTypeUint16, Data: uint16(0)}},
	}, []TensorData{{Name: "matrix.weight", Shape: []uint64{32, 2}, Type: DTypeF32, Data: bytes.NewReader(float32Bytes(values))}}, WriteOptions{})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "source.gguf")
	if err := os.WriteFile(path, source.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	file, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	computed := 0
	reader := &rewriteReader{file: file, tensor: file.Tensors[0], target: DTypeF32, transform: func(TensorInfo, []float32) error {
		computed++
		return nil
	}}
	data, err := io.ReadAll(reader)
	if err != nil || !bytes.Equal(data, float32Bytes(values)) {
		t.Fatalf("the rewritten tensor = %d bytes, %v; want the source's %d", len(data), err, len(values)*4)
	}
	if reader.encoded != nil {
		t.Error("the reader still holds its encoded bytes after its last byte was read")
	}
	if count, err := reader.Read(make([]byte, 1)); count != 0 || err != io.EOF || computed != 1 {
		t.Errorf("a read after the end gave %d bytes, %v, with the tensor computed %d times; want the end and one computation", count, err, computed)
	}
}
