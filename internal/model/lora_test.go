package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"llamacpp2go/internal/gguf"
)

func TestLoadLoRAValidatesProjectionAndEmbeddingPairs(t *testing.T) {
	directory := t.TempDir()
	basePath := filepath.Join(directory, "base.gguf")
	writeLoRAGGUF(t, basePath, nil, []gguf.TensorData{
		loRATensor("blk.0.attn_q.weight", []uint64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		loRATensor("token_embd.weight", []uint64{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8}),
	})
	adapterPath := filepath.Join(directory, "adapter.gguf")
	writeLoRAGGUF(t, adapterPath, loRAMetadata("llama"), []gguf.TensorData{
		loRATensor("blk.0.attn_q.weight.lora_a", []uint64{2, 1}, []float32{2, 3}),
		loRATensor("blk.0.attn_q.weight.lora_b", []uint64{1, 3}, []float32{4, 5, 6}),
		loRATensor("token_embd.weight.lora_a", []uint64{1, 4}, []float32{7, 8, 9, 10}),
		loRATensor("token_embd.weight.lora_b", []uint64{1, 2}, []float32{11, 12}),
	})
	base, err := gguf.Open(basePath)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	adapter, err := LoadLoRA(context.Background(), adapterPath, base, Spec{Architecture: "llama"})
	if err != nil {
		t.Fatal(err)
	}
	if adapter.Alpha != 4 || len(adapter.Weights) != 2 || !adapter.Weights["token_embd.weight"].Embedding {
		t.Fatalf("unexpected adapter: %+v", adapter)
	}
	if got := adapter.InvocationTokens; len(got) != 2 || got[0] != 17 || got[1] != 23 {
		t.Fatalf("invocation tokens = %v", got)
	}
}

func TestLoadLoRARejectsIncompletePair(t *testing.T) {
	directory := t.TempDir()
	basePath := filepath.Join(directory, "base.gguf")
	writeLoRAGGUF(t, basePath, nil, []gguf.TensorData{
		loRATensor("output.weight", []uint64{2, 3}, make([]float32, 6)),
	})
	adapterPath := filepath.Join(directory, "adapter.gguf")
	writeLoRAGGUF(t, adapterPath, loRAMetadata("llama"), []gguf.TensorData{
		loRATensor("output.weight.lora_a", []uint64{2, 1}, []float32{1, 2}),
	})
	base, err := gguf.Open(basePath)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	_, err = LoadLoRA(context.Background(), adapterPath, base, Spec{Architecture: "llama"})
	if err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("error = %v", err)
	}
}

func loRAMetadata(architecture string) []gguf.Metadata {
	return []gguf.Metadata{
		{Key: "general.type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "adapter"}},
		{Key: "general.architecture", Value: gguf.Value{Type: gguf.ValueTypeString, Data: architecture}},
		{Key: "adapter.type", Value: gguf.Value{Type: gguf.ValueTypeString, Data: "lora"}},
		{Key: "adapter.lora.alpha", Value: gguf.Value{Type: gguf.ValueTypeFloat32, Data: float32(4)}},
		{Key: "adapter.alora.invocation_tokens", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeUint32, Data: []uint32{17, 23}}},
	}
}

func loRATensor(name string, shape []uint64, values []float32) gguf.TensorData {
	var data bytes.Buffer
	_ = binary.Write(&data, binary.LittleEndian, values)
	return gguf.TensorData{Name: name, Shape: shape, Type: gguf.DTypeF32, Data: bytes.NewReader(data.Bytes())}
}

func writeLoRAGGUF(t *testing.T, path string, metadata []gguf.Metadata, tensors []gguf.TensorData) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gguf.Write(file, metadata, tensors, gguf.WriteOptions{}); err != nil {
		_ = file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
