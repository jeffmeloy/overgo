package safetensors

import (
	"io"
	"math"
	"path/filepath"
	"testing"
)

// TestSaveRoundTrip writes an artifact with Save and reads every tensor back
// through OpenSource/F32Reader, asserting byte-exact recovery of values,
// shapes, and metadata. This pins the writer to the reader it must feed.
func TestSaveRoundTrip(t *testing.T) {
	tensors := map[string][]float32{
		"model.embed_tokens.weight": {0, 1, 2, 3, 4, 5},                  // [3,2]
		"model.norm.weight":         {1, 1, 1, 1},                        // [4]
		"a.negatives_and_frac":      {-1.5, 0.25, math.MaxFloat32, -0.0}, // [2,2]
	}
	shapes := map[string][]int{
		"model.embed_tokens.weight": {3, 2},
		"model.norm.weight":         {4},
		"a.negatives_and_frac":      {2, 2},
	}
	metadata := map[string]string{"format": "pt", "trainer": "overgo"}

	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	if err := Save(path, tensors, shapes, metadata); err != nil {
		t.Fatalf("Save: %v", err)
	}

	source, err := OpenSource(dir)
	if err != nil {
		t.Fatalf("OpenSource: %v", err)
	}
	defer source.Close()

	if len(source.Tensors) != len(tensors) {
		t.Fatalf("tensor count = %d, want %d", len(source.Tensors), len(tensors))
	}
	for name, want := range tensors {
		tensor, ok := source.Tensors[name]
		if !ok {
			t.Fatalf("tensor %q missing", name)
		}
		wantShape := shapes[name]
		if len(tensor.Shape) != len(wantShape) {
			t.Fatalf("%q rank = %d, want %d", name, len(tensor.Shape), len(wantShape))
		}
		for i, dim := range wantShape {
			if tensor.Shape[i] != uint64(dim) {
				t.Fatalf("%q shape[%d] = %d, want %d", name, i, tensor.Shape[i], dim)
			}
		}
		reader, err := F32Reader(tensor)
		if err != nil {
			t.Fatalf("%q F32Reader: %v", name, err)
		}
		buf := make([]byte, len(want)*4)
		if _, err := io.ReadFull(reader, buf); err != nil {
			t.Fatalf("%q read: %v", name, err)
		}
		for i, w := range want {
			bits := uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 | uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24
			if got := math.Float32frombits(bits); got != w && !(math.IsNaN(float64(got)) && math.IsNaN(float64(w))) {
				t.Fatalf("%q[%d] = %v, want %v", name, i, got, w)
			}
		}
	}

	meta, ok := source.Metadata["model.safetensors"]
	if !ok {
		t.Fatalf("metadata missing for shard; have %v", source.Metadata)
	}
	if meta["trainer"] != "overgo" || meta["format"] != "pt" {
		t.Fatalf("metadata = %v, want trainer=overgo format=pt", meta)
	}
}

func TestSaveRejectsInvalidTensorGeometry(t *testing.T) {
	tensors := map[string][]float32{"tensor": {1}}
	for name, shape := range map[string][]int{
		"missing":  nil,
		"negative": {-1},
		"mismatch": {2},
		"overflow": {math.MaxInt, 3},
	} {
		t.Run(name, func(t *testing.T) {
			shapes := map[string][]int{}
			if name != "missing" {
				shapes["tensor"] = shape
			}
			if err := Save(filepath.Join(t.TempDir(), "invalid.safetensors"), tensors, shapes, nil); err == nil {
				t.Fatal("invalid tensor geometry accepted")
			}
		})
	}
}
