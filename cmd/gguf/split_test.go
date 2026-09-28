package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
)

func TestParseSize(t *testing.T) {
	for input, want := range map[string]uint64{
		"":    0,
		"1":   1,
		"2K":  2 << 10,
		"3m":  3 << 20,
		"4 G": 4 << 30,
	} {
		got, err := parseSize(input)
		if err != nil || got != want {
			t.Fatalf("parseSize(%q) = %d, %v; want %d", input, got, err, want)
		}
	}
	for _, input := range []string{"0", "-1", "x", "18446744073709551615G"} {
		if _, err := parseSize(input); err == nil {
			t.Fatalf("parseSize(%q) succeeded", input)
		}
	}
}

func TestSplitCreatesOpenableShards(t *testing.T) {
	directory := t.TempDir()
	inputPath := filepath.Join(directory, "input.gguf")
	input, err := os.Create(inputPath)
	if err != nil {
		t.Fatal(err)
	}
	tensors := make([]gguf.TensorData, 3)
	for index := range tensors {
		tensors[index] = gguf.TensorData{
			Name:  string(rune('a' + index)),
			Shape: []uint64{1},
			Type:  gguf.DTypeF32,
			Data:  bytes.NewReader([]byte{byte(index), 0, 0, 0}),
		}
	}
	err = gguf.Write(input, nil, tensors, gguf.WriteOptions{})
	if closeErr := input.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	prefix := filepath.Join(directory, "output")
	if err := split(inputPath, prefix, gguf.SplitOptions{MaxTensors: 1}); err != nil {
		t.Fatal(err)
	}
	first := prefix + "-00001-of-00003.gguf"
	file, err := gguf.Open(first)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if file.SplitCount != 3 || len(file.Tensors) != 3 {
		t.Fatalf("split/tensors = %d/%d, want 3/3", file.SplitCount, len(file.Tensors))
	}
	if err := split(inputPath, prefix, gguf.SplitOptions{MaxTensors: 1}); err == nil {
		t.Fatal("existing shard set was overwritten")
	}
}
