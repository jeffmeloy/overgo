package pytorchzip

import (
	"archive/zip"
	"math"
	"strings"
	"testing"
)

func TestMetadataSizeBoundBeforeConversion(t *testing.T) {
	for _, size := range []uint64{uint64(maxPickleBytes) + 1, uint64(math.MaxInt64) + 1, math.MaxUint64} {
		entry := &zip.File{FileHeader: zip.FileHeader{Name: "archive/data.pkl", UncompressedSize64: size}}
		data, err := readZipEntryBytes(entry, maxPickleBytes)
		if err == nil || data != nil || !strings.Contains(err.Error(), "too large") {
			t.Fatalf("metadata size %d: %v", size, err)
		}
	}
}
