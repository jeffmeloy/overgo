package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testcheck"
)

func TestMergeWritesCanonicalCopyWithoutOverwriting(t *testing.T) {
	directory := t.TempDir()
	inputPath := filepath.Join(directory, "input.gguf")
	outputPath := filepath.Join(directory, "output.gguf")
	input, err := os.Create(inputPath)
	testcheck.NoError(t, err)
	payload := []byte{0, 0, 0x80, 0x3f}
	err = gguf.Write(
		input,
		[]gguf.Metadata{{
			Key: "general.name",
			Value: gguf.Value{
				Type: gguf.ValueTypeString,
				Data: "fixture",
			},
		}},
		[]gguf.TensorData{{
			Name:  "weight",
			Shape: []uint64{1},
			Type:  gguf.DTypeF32,
			Data:  bytes.NewReader(payload),
		}},
		gguf.WriteOptions{},
	)
	if closeErr := input.Close(); err == nil {
		err = closeErr
	}
	testcheck.NoError(t, err)

	testcheck.NoError(t, merge(inputPath, outputPath))
	output := testcheck.Must(gguf.Open(outputPath))(t)
	defer output.Close()
	got := make([]byte, len(payload))
	testcheck.NoError(t, output.ReadTensorData(output.Tensors[0], got))
	if !bytes.Equal(got, payload) {
		t.Fatalf("payload = %x, want %x", got, payload)
	}
	if err := merge(inputPath, outputPath); err == nil {
		t.Fatal("existing output was overwritten")
	}
}

func TestMergeRejectsInputOutputIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "same.gguf")
	if err := merge(path, path); err == nil {
		t.Fatal("identical input/output paths were accepted")
	}
}
