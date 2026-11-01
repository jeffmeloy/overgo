package hfgguf

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/safetensors"
)

type countedTensorSource struct {
	data  []byte
	reads int
}

func (r *countedTensorSource) ReadAt(p []byte, offset int64) (int, error) {
	r.reads++
	return bytes.NewReader(r.data).ReadAt(p, offset)
}

func streamMapping(t *testing.T, source *countedTensorSource, name string) tensorMapping {
	t.Helper()
	tensor, err := safetensors.NewTensor(name, "F32", []uint64{4}, source, 0, 16)
	if err != nil {
		t.Fatal(err)
	}
	return tensorMapping{name: name, tensor: tensor, shape: []uint64{4}, storage: gguf.DTypeF32, transforms: []TensorTransform{{Operation: transformAddOne}}}
}

func TestMappedTensorStreaming(t *testing.T) {
	t.Run("read-on-demand", func(t *testing.T) {
		first, second := &countedTensorSource{data: make([]byte, 16)}, &countedTensorSource{data: make([]byte, 16)}
		tensors, err := mappedTensorData([]tensorMapping{streamMapping(t, first, "first"), streamMapping(t, second, "second")})
		if err != nil {
			t.Fatal(err)
		}
		if first.reads != 0 || second.reads != 0 {
			t.Fatal("conversion materialized unread tensors")
		}
		if n, err := tensors[0].Data.Read(nil); n != 0 || err != nil || first.reads != 0 {
			t.Fatal("empty read consumed a tensor")
		}
		output := make([]byte, 16)
		if _, err := io.ReadFull(tensors[0].Data, output); err != nil {
			t.Fatal(err)
		}
		completed := tensors[0].Data.(*transformedPayload)
		if completed.data != nil || completed.source != nil || completed.transforms != nil {
			t.Fatal("completed tensor retains payload or source")
		}
		if first.reads == 0 || second.reads != 0 {
			t.Fatal("reading one tensor consumed another")
		}
		want := bytes.Repeat([]byte{0, 0, 128, 63}, 4)
		if !bytes.Equal(output, want) {
			t.Fatal("transformed bytes changed")
		}
		if n, err := tensors[0].Data.Read(output); n != 0 || !errors.Is(err, io.EOF) {
			t.Fatalf("finished read = %d, %v", n, err)
		}
		if n, err := io.Copy(io.Discard, tensors[1].Data); n != 16 || err != nil {
			t.Fatalf("second payload = %d, %v", n, err)
		}
	})
	t.Run("source-failure-is-sticky", func(t *testing.T) {
		source := &countedTensorSource{data: []byte{1, 2, 3}}
		tensors, err := mappedTensorData([]tensorMapping{streamMapping(t, source, "short")})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.ReadAll(tensors[0].Data); err == nil {
			t.Fatal("accepted truncated transformed source")
		}
		reads := source.reads
		if _, err := io.ReadAll(tensors[0].Data); err == nil || source.reads != reads {
			t.Fatal("failed tensor was retried or accepted")
		}
	})
	t.Run("writer-failure-leaves-payloads-unread", func(t *testing.T) {
		source := &countedTensorSource{data: make([]byte, 16)}
		tensors, err := mappedTensorData([]tensorMapping{streamMapping(t, source, "weight")})
		if err != nil {
			t.Fatal(err)
		}
		sentinel := errors.New("destination unavailable")
		if err := gguf.Write(refusingTensorWriter{sentinel}, nil, tensors, gguf.WriteOptions{}); !errors.Is(err, sentinel) {
			t.Fatalf("writer failure = %v", err)
		}
		if source.reads != 0 {
			t.Fatal("failed output caused weight reads")
		}
	})
	t.Run("failed-file-is-removed", func(t *testing.T) {
		broken, later := &countedTensorSource{data: make([]byte, 4)}, &countedTensorSource{data: make([]byte, 16)}
		tensors, err := mappedTensorData([]tensorMapping{streamMapping(t, broken, "broken"), streamMapping(t, later, "later")})
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "incomplete.gguf")
		if err := gguf.WriteFileExclusive(path, nil, tensors, gguf.WriteOptions{}); err == nil {
			t.Fatal("published truncated conversion")
		}
		if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("partial output survived: %v", err)
		}
		if later.reads != 0 {
			t.Fatal("conversion continued after a failed tensor")
		}
	})
	t.Run("transform-failure-is-sticky", func(t *testing.T) {
		source := &countedTensorSource{data: make([]byte, 16)}
		mapping := streamMapping(t, source, "invalid")
		mapping.transforms = []TensorTransform{{Operation: "unknown"}}
		tensors, err := mappedTensorData([]tensorMapping{mapping})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.ReadAll(tensors[0].Data); err == nil {
			t.Fatal("accepted unknown transform")
		}
		reads := source.reads
		if _, err := io.ReadAll(tensors[0].Data); err == nil || source.reads != reads {
			t.Fatal("failed transformation retried or succeeded")
		}
		if tensors[0].Data.(*transformedPayload).data != nil {
			t.Fatal("failed transform retains payload")
		}
	})
	t.Run("gguf-roundtrip", func(t *testing.T) {
		first, second := &countedTensorSource{data: make([]byte, 16)}, &countedTensorSource{data: make([]byte, 16)}
		tensors, err := mappedTensorData([]tensorMapping{streamMapping(t, first, "first"), streamMapping(t, second, "second")})
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "complete.gguf")
		if err := gguf.WriteFileExclusive(path, nil, tensors, gguf.WriteOptions{}); err != nil {
			t.Fatal(err)
		}
		file, err := gguf.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		if len(file.Tensors) != len(tensors) {
			t.Fatal("lost a tensor")
		}
		for i, info := range file.Tensors {
			output := make([]byte, info.Size)
			if err := file.ReadTensorData(info, output); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(output, bytes.Repeat([]byte{0, 0, 128, 63}, 4)) {
				t.Fatal("GGUF tensor bytes changed")
			}
			if tensors[i].Data.(*transformedPayload).data != nil {
				t.Fatal("writer retained a completed tensor")
			}
		}
	})
}

type refusingTensorWriter struct{ err error }

func (w refusingTensorWriter) Write([]byte) (int, error) { return 0, w.err }
