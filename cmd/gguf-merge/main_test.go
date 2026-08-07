package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
)

func TestMergeWritesCanonicalCopyWithoutOverwriting(t *testing.T) {
	directory := t.TempDir()
	inputPath := filepath.Join(directory, "input.gguf")
	outputPath := filepath.Join(directory, "output.gguf")
	input, err := os.Create(inputPath)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}

	if err := merge(inputPath, outputPath); err != nil {
		t.Fatal(err)
	}
	output, err := gguf.Open(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	got := make([]byte, len(payload))
	if err := output.ReadTensorData(output.Tensors[0], got); err != nil {
		t.Fatal(err)
	}
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
