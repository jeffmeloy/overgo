package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/repodb"
)

func writeModelFixture(t *testing.T, path string) {
	t.Helper()
	values := make([]byte, 512*4)
	for i := 0; i < 512; i++ {
		binary.LittleEndian.PutUint32(values[i*4:], math.Float32bits(float32(i-256)))
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(f, nil, []gguf.TensorData{
		{Name: "blk.0.weight", Shape: []uint64{512}, Type: gguf.DTypeF32, Data: bytes.NewReader(values)},
	}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCharacterizeAndStorePersistsWithLineage(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.gguf")
	writeModelFixture(t, modelPath)

	storeDir := filepath.Join(dir, "store")
	if err := os.MkdirAll(storeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(storeDir)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	file, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	ctx := context.Background()
	policy := modelartifact.MeasurementPolicy{MaxSamplesPerTensor: 256, MaxReadBytes: 1 << 20}
	id, err := characterizeAndStore(ctx, store, file, policy)
	if err != nil {
		t.Fatal(err)
	}

	content, ok, err := store.Content(ctx, id)
	if err != nil || !ok {
		t.Fatalf("stored measurement not found: ok=%v err=%v", ok, err)
	}
	if content.Descriptor.MediaType != modelartifact.TensorMeasurementMediaType {
		t.Errorf("stored media type = %q, want %q", content.Descriptor.MediaType, modelartifact.TensorMeasurementMediaType)
	}
	parents, err := store.Parents(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if len(parents) == 0 {
		t.Fatal("measurement has no lineage parent (inventory)")
	}

	// Idempotent: re-characterizing the same model returns the same identity.
	id2, err := characterizeAndStore(ctx, store, file, policy)
	if err != nil {
		t.Fatalf("re-characterize failed: %v", err)
	}
	if id2 != id {
		t.Errorf("re-characterize id = %s, want %s", id2, id)
	}
}
