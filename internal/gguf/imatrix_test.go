package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadGGUFImportanceMatrixNormalizesExpertCounts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "imatrix.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	metadata := []Metadata{
		{Key: "imatrix.datasets", Value: Value{Type: ValueTypeArray, ArrayType: ValueTypeString, Data: []string{"fixture"}}},
		{Key: "imatrix.chunk_count", Value: Value{Type: ValueTypeUint32, Data: uint32(3)}},
		{Key: "imatrix.chunk_size", Value: Value{Type: ValueTypeUint32, Data: uint32(512)}},
	}
	sums := []float32{2, 4, 6, 8, 10, 12, 14, 16}
	counts := []float32{2, 0}
	err = Write(file, metadata, []TensorData{
		{Name: "blk.0.weight.in_sum2", Shape: []uint64{4, 2}, Type: DTypeF32, Data: bytes.NewReader(float32Bytes(sums))},
		{Name: "blk.0.weight.counts", Shape: []uint64{2}, Type: DTypeF32, Data: bytes.NewReader(float32Bytes(counts))},
	}, WriteOptions{})
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	matrix, err := LoadImportanceMatrix(path)
	if err != nil {
		t.Fatal(err)
	}
	if matrix.Legacy || matrix.ChunkCount != 3 || matrix.ChunkSize != 512 ||
		!reflect.DeepEqual(matrix.Datasets, []string{"fixture"}) {
		t.Fatalf("metadata = %+v", matrix)
	}
	want := []float32{1, 2, 3, 4, 1, 1, 1, 1}
	if got := matrix.Entries["blk.0.weight"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("weights = %v, want %v", got, want)
	}
}

func TestLoadLegacyImportanceMatrix(t *testing.T) {
	path := filepath.Join(t.TempDir(), "imatrix.dat")
	var data bytes.Buffer
	write := func(value any) { _ = binary.Write(&data, binary.LittleEndian, value) }
	name := "blk.0.weight"
	write(int32(1))
	write(int32(len(name)))
	_, _ = data.WriteString(name)
	write(int32(2))
	write(int32(4))
	write([]float32{2, 4, 6, 8})
	write(int32(5))
	dataset := "legacy-fixture"
	write(int32(len(dataset)))
	_, _ = data.WriteString(dataset)
	if err := os.WriteFile(path, data.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	matrix, err := LoadImportanceMatrix(path)
	if err != nil {
		t.Fatal(err)
	}
	if !matrix.Legacy || matrix.ChunkCount != 5 ||
		!reflect.DeepEqual(matrix.Datasets, []string{dataset}) ||
		!reflect.DeepEqual(matrix.Entries[name], []float32{1, 2, 3, 4}) {
		t.Fatalf("matrix = %+v", matrix)
	}
}
