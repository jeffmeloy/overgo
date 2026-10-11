package export

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

// fileBytes is the size of a file on disk.
func fileBytes(t *testing.T, file string) uint64 {
	t.Helper()
	info, err := os.Stat(file)
	if err != nil {
		t.Fatal(err)
	}
	return uint64(info.Size())
}

// TestSizesAreTheBytesWritten holds each size function to the model file its
// writer writes, to the byte: an f16 export of the hermetic model, a q4_0
// export of a matrix whose rows are one quantization block wide (the hermetic
// model's rows are narrower than any block, so it would say nothing about
// block storage), and a merge. Each export's size differs from its source's,
// so an echo of the source does not pass. What a writer refuses before it
// writes, its size function refuses too.
func TestSizesAreTheBytesWritten(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	hermetic := testutil.HermeticLlamaGGUF(t, 64)
	blocks := testutil.TempGGUF(t, "blocks.gguf", nil, []gguf.TensorData{testutil.GGUFTensorF32("matrix.weight", []uint64{32, 2}, 1)})
	for _, leg := range []struct{ source, typeName string }{{hermetic, "f16"}, {blocks, "q4_0"}} {
		size, err := GGUFSize(leg.source, leg.typeName)
		if err != nil {
			t.Fatalf("the size of a %s export: %v", leg.typeName, err)
		}
		file, err := GGUF(ctx, leg.source, root, leg.typeName, leg.typeName)
		if err != nil {
			t.Fatal(err)
		}
		if wrote, source := fileBytes(t, file), fileBytes(t, leg.source); size != wrote || size == source {
			t.Fatalf("a %s export was sized at %d bytes and wrote %d, from a source of %d", leg.typeName, size, wrote, source)
		}
	}
	tuned, scales := []string{hermetic}, []float32{0.5}
	size, err := MergeGGUFSize(hermetic, tuned, scales)
	if err != nil {
		t.Fatalf("the size of a merge: %v", err)
	}
	file, err := MergeGGUF(ctx, hermetic, tuned, scales, root, "merged")
	if err != nil {
		t.Fatal(err)
	}
	if wrote := fileBytes(t, file); size != wrote {
		t.Fatalf("a merge was sized at %d bytes and wrote %d", size, wrote)
	}
	if size, err := GGUFSize(hermetic, "iq2_xxs"); err == nil || !strings.Contains(err.Error(), "without an importance matrix") {
		t.Fatalf("a type that needs an importance matrix was sized at %d bytes: %v", size, err)
	}
	if size, err := MergeGGUFSize(hermetic, tuned, nil); err == nil || !strings.Contains(err.Error(), "needs its scale") {
		t.Fatalf("a merge of a model with no scale was sized at %d bytes: %v", size, err)
	}
}

// unreadable is a tensor's bytes that nothing may read.
type unreadable struct{}

// Read fails, so a write that reads the tensor fails with it.
func (unreadable) Read([]byte) (int, error) { return 0, errors.New("a tensor was read") }

// TestSizingReadsNoTensor holds the counting destination to the cost a size
// may have: the GGUF writer writes a file of tensors nothing can read into it
// and it counts the bytes a real write of real tensors makes, padding
// included. A tensor longer than the writer copies at once, as a real model's
// are, is counted whole across its copies. A source the writer did not bound
// it counts by reading, so the count stays right if the copy ever hands it
// another kind of reader.
func TestSizingReadsNoTensor(t *testing.T) {
	t.Parallel()
	// The GGUF writer copies a tensor this many bytes at a time, and an F32
	// element is four bytes.
	const copied, elementBytes = uint64(1 << 30), 4
	matrix, vector := []uint64{32, 2}, []uint64{3}
	var written bytes.Buffer
	if err := gguf.Write(&written, nil, []gguf.TensorData{
		testutil.GGUFTensorF32("matrix.weight", matrix, 1), testutil.GGUFTensorF32("vector.weight", vector, 2),
	}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	sized := func(vector []uint64) uint64 {
		t.Helper()
		var size sizer
		if err := gguf.Write(&size, nil, []gguf.TensorData{
			{Name: "matrix.weight", Shape: matrix, Type: gguf.DTypeF32, Data: unreadable{}},
			{Name: "vector.weight", Shape: vector, Type: gguf.DTypeF32, Data: unreadable{}},
		}, gguf.WriteOptions{}); err != nil {
			t.Fatalf("sizing unreadable tensors: %v", err)
		}
		return uint64(size)
	}
	if size := sized(vector); size != uint64(written.Len()) {
		t.Fatalf("sizing unreadable tensors counted %d bytes; a real write makes %d", size, written.Len())
	}
	// The same file with its vector one copy longer: the header is as long, and
	// the storage is longer by that copy exactly.
	if size := sized([]uint64{vector[0] + copied/elementBytes}); size != uint64(written.Len())+copied {
		t.Fatalf("a tensor one copy of %d bytes longer was counted as %d bytes, want %d", copied, size, uint64(written.Len())+copied)
	}
	var read sizer
	if count, err := read.ReadFrom(strings.NewReader("tensor")); err != nil || count != int64(len("tensor")) || uint64(read) != uint64(count) {
		t.Fatalf("an unbounded source of %d bytes was counted as %d (%d kept), %v", len("tensor"), count, read, err)
	}
}
