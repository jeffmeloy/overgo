package pytorchzip

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/tensor/dtype"
)

// pickleWriter builds a minimal torch state_dict pickle stream by hand.
type pickleWriter struct{ buf bytes.Buffer }

func (w *pickleWriter) op(b byte)    { w.buf.WriteByte(b) }
func (w *pickleWriter) raw(b []byte) { w.buf.Write(b) }
func (w *pickleWriter) str(s string) { w.op('X'); w.u32(uint32(len(s))); w.buf.WriteString(s) }
func (w *pickleWriter) global(m, n string) {
	w.op('c')
	w.buf.WriteString(m + "\n" + n + "\n")
}
func (w *pickleWriter) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.raw(b[:])
}
func (w *pickleWriter) int32v(v int32) { w.op('J'); w.u32(uint32(v)) }

// tensorEntry appends one named _rebuild_tensor_v2 REDUCE + dict pair items
// (caller wraps with MARK ... SETITEMS).
func (w *pickleWriter) tensorEntry(name, storageClass, storageKey string, storageSize, offset int64, shape, stride []int64) {
	w.str(name)
	w.global("torch._utils", "_rebuild_tensor_v2")
	w.op('(') // args
	// persistent id tuple: ("storage", Global, key, location, size)
	w.op('(')
	w.str("storage")
	w.global("torch", storageClass)
	w.str(storageKey)
	w.str("cpu")
	w.int32v(int32(storageSize))
	w.op('t')
	w.op('Q')
	w.int32v(int32(offset))
	w.op('(')
	for _, d := range shape {
		w.int32v(int32(d))
	}
	w.op('t')
	w.op('(')
	for _, s := range stride {
		w.int32v(int32(s))
	}
	w.op('t')
	w.op(0x89) // requires_grad = False
	w.op(')')  // backward hooks
	w.op('t')
	w.op('R')
}

func syntheticCheckpoint(t *testing.T, values []float32) (string, []byte) {
	t.Helper()
	var w pickleWriter
	w.op(0x80)
	w.op(2) // PROTO 2
	w.op('}')
	w.op('q')
	w.op(0)
	w.op('(')
	w.tensorEntry("w", "BFloat16Storage", "0", int64(len(values)), 0, []int64{2, 3}, []int64{3, 1})
	w.op('u')
	w.op('.')
	body := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(body[i*2:], dtype.Float32ToBF16(v))
	}
	var archive bytes.Buffer
	zw := zip.NewWriter(&archive)
	pkl, err := zw.CreateHeader(&zip.FileHeader{Name: "archive/data.pkl", Method: zip.Deflate})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pkl.Write(w.buf.Bytes()); err != nil {
		t.Fatal(err)
	}
	storage, err := zw.CreateHeader(&zip.FileHeader{Name: "archive/data/0", Method: zip.Store})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := storage.Write(body); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "synthetic.pth")
	if err := os.WriteFile(path, archive.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path, w.buf.Bytes()
}

func TestSyntheticCheckpointRoundTrip(t *testing.T) {
	values := []float32{-2.5, 0, 1.25, 255, -0.375, 8}
	path, pickle := syntheticCheckpoint(t, values)
	metas, err := ReadTensorMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(metas) != 1 || metas[0].Name != "w" || metas[0].DType != "BFloat16Storage" || metas[0].Numel != 6 {
		t.Fatalf("metas=%+v", metas)
	}
	bindings, err := CompileBindings(metas, []string{"w"})
	if err != nil {
		t.Fatal(err)
	}
	reader, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	got, err := reader.ReadBinding(bindings[0])
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range values {
		if got[i] != want { // all values BF16-exact by construction
			t.Fatalf("value[%d]=%g want %g", i, got[i], want)
		}
	}
	rows, err := reader.ReadTensorRows(bindings[0], []int{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	wantRows := []float32{255, -0.375, 8, -2.5, 0, 1.25}
	for i, want := range wantRows {
		if rows[i] != want {
			t.Fatalf("row value[%d]=%g want %g", i, rows[i], want)
		}
	}
	// Truncated pickle must refuse, not panic.
	for cut := 1; cut < len(pickle); cut += 7 {
		if _, err := ParseTensorMetadata(pickle[:cut]); err == nil {
			t.Fatalf("truncated pickle at %d parsed without error", cut)
		}
	}
}

func TestCompressedStorageRefused(t *testing.T) {
	values := []float32{1, 2, 3, 4, 5, 6}
	path, _ := syntheticCheckpoint(t, values)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Rebuild with a DEFLATED storage entry: reader must refuse (offset reads
	// require stored bodies).
	var archive bytes.Buffer
	zr, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(&archive)
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		dst, err := zw.CreateHeader(&zip.FileHeader{Name: f.Name, Method: zip.Deflate})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.Copy(dst, rc); err != nil {
			t.Fatal(err)
		}
		rc.Close()
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	compressed := filepath.Join(t.TempDir(), "compressed.pth")
	if err := os.WriteFile(compressed, archive.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(compressed); err == nil || !strings.Contains(err.Error(), "compressed") {
		t.Fatalf("compressed storage accepted: %v", err)
	}
}
