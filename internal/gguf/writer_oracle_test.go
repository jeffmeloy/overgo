package gguf

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testevidence"
)

func TestWriterAcceptedByPinnedLlamaCPPGGUFHash(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": requires pinned llama-gguf-hash")
	}
	oracle := os.Getenv("OVERGO_GGUF_HASH_ORACLE")
	if oracle == "" {
		t.Skip("set OVERGO_GGUF_HASH_ORACLE to pinned llama-gguf-hash")
	}
	path := filepath.Join(t.TempDir(), "writer-oracle.gguf")
	destination, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	payload := make([]byte, 18)
	for index := range payload {
		payload[index] = byte(index*23 + 7)
	}
	err = Write(
		destination,
		[]Metadata{
			{
				Key: "general.name",
				Value: Value{
					Type: ValueTypeString,
					Data: "Go writer oracle",
				},
			},
			{
				Key: "test.values",
				Value: Value{
					Type:      ValueTypeArray,
					ArrayType: ValueTypeInt32,
					Data:      []int32{-1, 0, 1},
				},
			},
		},
		[]TensorData{{
			Name:  "test.weight",
			Shape: []uint64{32},
			Type:  DTypeQ4_0,
			Data:  bytes.NewReader(payload),
		}},
		WriteOptions{},
	)
	if closeErr := destination.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	command := exec.Command(oracle, "--sha256", path)
	command.Dir = filepath.Dir(oracle)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("pinned llama-gguf-hash rejected writer output: %v\n%s", err, output)
	}
	if !strings.Contains(string(output), "test.weight") {
		t.Fatalf("pinned llama-gguf-hash output omitted tensor:\n%s", output)
	}
}

func TestQuantizerOutputAcceptedByPinnedLlamaCPPGGUFHash(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": requires pinned llama-gguf-hash")
	}
	oracle := os.Getenv("OVERGO_GGUF_HASH_ORACLE")
	if oracle == "" {
		t.Skip("set OVERGO_GGUF_HASH_ORACLE to pinned llama-gguf-hash")
	}
	values := quantizeFixtureValues(64)
	var sourceData bytes.Buffer
	if err := Write(
		&sourceData,
		nil,
		[]TensorData{{
			Name:  "test.weight",
			Shape: []uint64{32, 2},
			Type:  DTypeF32,
			Data:  bytes.NewReader(float32Bytes(values)),
		}},
		WriteOptions{},
	); err != nil {
		t.Fatal(err)
	}
	source, err := Parse(
		bytes.NewReader(sourceData.Bytes()),
		uint64(sourceData.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}
	var outputData bytes.Buffer
	if _, err := source.QuantizeTo(
		&outputData,
		DTypeQ4_0,
		QuantizeOptions{},
	); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "quantizer-oracle.gguf")
	if err := os.WriteFile(path, outputData.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	command := exec.Command(oracle, "--sha256", path)
	command.Dir = filepath.Dir(oracle)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("pinned llama-gguf-hash rejected quantizer output: %v\n%s", err, output)
	}
	if !strings.Contains(string(output), "test.weight") {
		t.Fatalf("pinned llama-gguf-hash output omitted tensor:\n%s", output)
	}
}

func TestSplitWriterAcceptedByPinnedLlamaCPPMerge(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": requires pinned llama-gguf-split")
	}
	oracle := os.Getenv("OVERGO_GGUF_SPLIT_ORACLE")
	if oracle == "" {
		t.Skip("set OVERGO_GGUF_SPLIT_ORACLE to pinned llama-gguf-split")
	}
	var encoded bytes.Buffer
	tensors := make([]TensorData, 3)
	for index := range tensors {
		tensors[index] = TensorData{
			Name:  fmt.Sprintf("weight.%d", index),
			Shape: []uint64{1},
			Type:  DTypeF32,
			Data:  bytes.NewReader([]byte{byte(index + 1), 0, 0, 0}),
		}
	}
	if err := Write(&encoded, nil, tensors, WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	source, err := Parse(
		bytes.NewReader(encoded.Bytes()),
		uint64(encoded.Len()),
		DefaultOptions(),
	)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	prefix := filepath.Join(directory, "go-split")
	err = source.WriteSplit(
		func(index, count uint16) (io.WriteCloser, error) {
			return os.Create(formatSplitPath(prefix, index, count))
		},
		SplitOptions{MaxTensors: 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	first := formatSplitPath(prefix, 0, 3)
	mergedPath := filepath.Join(directory, "upstream-merged.gguf")
	command := exec.Command(oracle, "--merge", first, mergedPath)
	command.Dir = filepath.Dir(oracle)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("pinned llama-gguf-split rejected Go shards: %v\n%s", err, output)
	}
	merged, err := Open(mergedPath)
	if err != nil {
		t.Fatal(err)
	}
	defer merged.Close()
	if len(merged.Tensors) != len(tensors) {
		t.Fatalf("upstream merged %d tensors, want %d", len(merged.Tensors), len(tensors))
	}
}
