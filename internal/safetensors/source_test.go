package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenSourceUsesShardIndexAndReadsTensor(t *testing.T) {
	directory := t.TempDir()
	writeShard(t, filepath.Join(directory, "model-1.safetensors"), map[string]testTensor{
		"weight": {dataType: "F32", shape: []uint64{2}, data: []byte{1, 2, 3, 4, 5, 6, 7, 8}},
	})
	writeShard(t, filepath.Join(directory, "adapter.safetensors"), map[string]testTensor{
		"ignored": {dataType: "U8", shape: []uint64{1}, data: []byte{9}},
	})
	writeIndex(t, directory, map[string]string{"weight": "model-1.safetensors"})

	source, err := OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if names := source.Names(); len(names) != 1 || names[0] != "weight" {
		t.Fatalf("names = %v", names)
	}
	tensor := source.Tensors["weight"]
	data, err := io.ReadAll(tensor.Reader())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, []byte{1, 2, 3, 4, 5, 6, 7, 8}) {
		t.Fatalf("payload = %v", data)
	}
}

func TestSourceIntShapesUsesHostRepresentability(t *testing.T) {
	source := &Source{Tensors: map[string]Tensor{
		"matrix": {Shape: []uint64{2, 3}},
	}}
	shapes, err := source.IntShapes()
	if err != nil {
		t.Fatal(err)
	}
	if got := shapes["matrix"]; len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Fatalf("shape = %v", got)
	}
	source.Tensors["matrix"] = Tensor{Shape: []uint64{2, 0}}
	if _, err := source.IntShapes(); err == nil {
		t.Fatal("empty dimension accepted")
	}
}

func TestSourceSnapshotOwnsCatalogMetadata(t *testing.T) {
	directory := t.TempDir()
	writeShard(t, filepath.Join(directory, "model.safetensors"), map[string]testTensor{
		"weight": {dataType: "F32", shape: []uint64{1}, data: []byte{0, 0, 128, 63}},
	})
	source, err := OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	snapshot := source.Snapshot()
	original := source.Tensors["weight"]
	original.Shape[0] = 0
	source.Tensors["weight"] = original
	if got := snapshot.Tensors["weight"].Shape[0]; got != 1 {
		t.Fatalf("snapshot shape=%d want 1", got)
	}
	var payload [4]byte
	if _, err := snapshot.Tensors["weight"].ReadAt(payload[:], 0); err != nil {
		t.Fatal(err)
	}
	if payload != [4]byte{0, 0, 128, 63} {
		t.Fatalf("snapshot payload=%v", payload)
	}
}

func TestOpenSourceUsesDiffusersShardIndex(t *testing.T) {
	directory := t.TempDir()
	shard := "diffusion_pytorch_model-00001-of-00001.safetensors"
	writeShard(t, filepath.Join(directory, shard), map[string]testTensor{
		"weight": {dataType: "U8", shape: []uint64{1}, data: []byte{7}},
	})
	writeNamedIndex(t, directory, "diffusion_pytorch_model.safetensors.index.json", map[string]string{"weight": shard})

	source, err := OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if !source.Indexed() || len(source.Names()) != 1 || source.Names()[0] != "weight" {
		t.Fatalf("indexed/names = %t/%v", source.Indexed(), source.Names())
	}
}

func TestOpenSourceRejectsInvalidRepositories(t *testing.T) {
	for _, test := range []struct {
		name  string
		build func(*testing.T, string)
		want  string
	}{
		{
			name: "escaping shard",
			build: func(t *testing.T, directory string) {
				writeIndex(t, directory, map[string]string{"weight": "../outside.safetensors"})
			},
			want: "escapes repository",
		},
		{
			name: "index mismatch",
			build: func(t *testing.T, directory string) {
				writeShard(t, filepath.Join(directory, "model.safetensors"), map[string]testTensor{
					"actual": {dataType: "U8", shape: []uint64{1}, data: []byte{1}},
				})
				writeIndex(t, directory, map[string]string{"declared": "model.safetensors"})
			},
			want: "tensor \"declared\" is missing",
		},
		{
			name: "overlap",
			build: func(t *testing.T, directory string) {
				header := map[string]tensorHeader{
					"a": {DType: "U8", Shape: []uint64{2}, DataOffsets: []uint64{0, 2}},
					"b": {DType: "U8", Shape: []uint64{2}, DataOffsets: []uint64{1, 3}},
				}
				writeRawShard(t, filepath.Join(directory, "model.safetensors"), header, []byte{1, 2, 3})
			},
			want: "overlaps",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			directory := t.TempDir()
			test.build(t, directory)
			_, err := OpenSource(directory)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestOpenSourceEnforcesHeaderLimit(t *testing.T) {
	directory := t.TempDir()
	writeShard(t, filepath.Join(directory, "model.safetensors"), map[string]testTensor{
		"weight": {dataType: "U8", shape: []uint64{1}, data: []byte{1}},
	})
	limits := DefaultLimits()
	limits.MaxHeaderBytes = 1
	if _, err := OpenSourceWithLimits(directory, limits); err == nil || !strings.Contains(err.Error(), "header size") {
		t.Fatalf("error = %v", err)
	}
}

func TestNewTensorValidatesAndBoundsReads(t *testing.T) {
	reader := bytes.NewReader([]byte{0, 1, 2, 3, 4})
	tensor, err := NewTensor("x", "U8", []uint64{3}, reader, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	buffer := make([]byte, 2)
	if _, err := tensor.ReadAt(buffer, 1); err != nil || !bytes.Equal(buffer, []byte{2, 3}) {
		t.Fatalf("read = %v, %v", buffer, err)
	}
	if _, err := tensor.ReadAt(buffer, 2); err != io.ErrUnexpectedEOF {
		t.Fatalf("overflow error = %v", err)
	}
}

type testTensor struct {
	dataType string
	shape    []uint64
	data     []byte
}

func writeShard(t *testing.T, path string, tensors map[string]testTensor) {
	t.Helper()
	header := make(map[string]tensorHeader, len(tensors))
	body := make([]byte, 0)
	for name, tensor := range tensors {
		start := uint64(len(body))
		body = append(body, tensor.data...)
		header[name] = tensorHeader{
			DType: tensor.dataType, Shape: tensor.shape,
			DataOffsets: []uint64{start, uint64(len(body))},
		}
	}
	writeRawShard(t, path, header, body)
}

func writeRawShard(t *testing.T, path string, header map[string]tensorHeader, body []byte) {
	t.Helper()
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8, 8+len(encoded)+len(body))
	binary.LittleEndian.PutUint64(data, uint64(len(encoded)))
	data = append(data, encoded...)
	data = append(data, body...)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeIndex(t *testing.T, directory string, weights map[string]string) {
	writeNamedIndex(t, directory, "model.safetensors.index.json", weights)
}

func writeNamedIndex(t *testing.T, directory, name string, weights map[string]string) {
	t.Helper()
	encoded, err := json.Marshal(shardIndex{WeightMap: weights})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, name), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}
