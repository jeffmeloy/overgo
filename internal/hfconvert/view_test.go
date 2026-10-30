package hfconvert

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

// TestAQwen35ViewIsTheBytesTheConverterWrites holds the view of a Qwen 3.5 directory to the file Convert writes from
// it: its image, header, tensors and padding, is the converted file byte for byte, read whole and read in pieces that
// cross every boundary, so a model served from its own files reads what its conversion would have written. The
// directory carries the parts the converter transforms beside tensors it keeps as they are; the view builds the first
// on read and reads the second in place. gguf.Open serves the directory as the same view.
func TestAQwen35ViewIsTheBytesTheConverterWrites(t *testing.T) {
	t.Parallel()
	directory := testutil.Qwen35Directory(t)
	converted := filepath.Join(t.TempDir(), "converted.gguf")
	if _, err := Convert(Options{Directory: directory, OutputPath: converted, Name: "fixture"}); err != nil {
		t.Fatal(err)
	}
	written, err := os.ReadFile(converted)
	if err != nil {
		t.Fatal(err)
	}
	file, err := View(directory, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	shards := file.Shards()
	if len(shards) != 1 || shards[0].Size != uint64(len(written)) {
		t.Fatalf("the view's image is %+v, want one of the converted file's %d bytes", shards, len(written))
	}
	whole := make([]byte, len(written))
	if _, err := shards[0].Source.ReadAt(whole, 0); err != nil {
		t.Fatal(err)
	}
	if at := firstDifference(whole, written); at >= 0 {
		t.Fatalf("the view read whole differs from the converted file at byte %d of %d", at, len(written))
	}
	pieces := make([]byte, 0, len(written))
	for offset := 0; offset < len(written); offset += 7 {
		piece := make([]byte, min(7, len(written)-offset))
		if _, err := shards[0].Source.ReadAt(piece, int64(offset)); err != nil {
			t.Fatal(err)
		}
		pieces = append(pieces, piece...)
	}
	if at := firstDifference(pieces, written); at >= 0 {
		t.Fatalf("the view read in pieces differs from the converted file at byte %d", at)
	}
	opened, err := gguf.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if opened.ViewDirectory() == "" || len(opened.Tensors) != len(file.Tensors) || len(opened.Metadata) != len(file.Metadata) {
		t.Errorf("gguf.Open of the directory = %d tensors, %d metadata, viewing %q; want the view of it", len(opened.Tensors), len(opened.Metadata), opened.ViewDirectory())
	}
}

// firstDifference is the first index where two byte slices differ, or -1.
func firstDifference(left, right []byte) int {
	if bytes.Equal(left, right) {
		return -1
	}
	for index := range min(len(left), len(right)) {
		if left[index] != right[index] {
			return index
		}
	}
	return min(len(left), len(right))
}
