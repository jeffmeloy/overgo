package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

func TestParseQuantizationType(t *testing.T) {
	for input, want := range map[string]dtype.Type{
		"Q4_0":    dtype.Q4_0,
		"q5-1":    dtype.Q5_1,
		" bf16":   dtype.BF16,
		"MXFP4":   dtype.MXFP4,
		"q6-k":    dtype.Q6K,
		"TQ1-0":   dtype.TQ1_0,
		"NVFP4":   dtype.NVFP4,
		"iq2-s":   dtype.IQ2S,
		"iq2-xxs": dtype.IQ2XXS,
		"iq2-xs":  dtype.IQ2XS,
		"iq1-s":   dtype.IQ1S,
		"iq1-m":   dtype.IQ1M,
		"iq3_s":   dtype.IQ3S,
		"iq4-xs":  dtype.IQ4XS,
	} {
		got, err := parseQuantizationType(input)
		if err != nil || got != want {
			t.Fatalf("parseQuantizationType(%q) = %s, %v; want %s",
				input, got, err, want)
		}
	}
	if _, err := parseQuantizationType("unknown"); err == nil {
		t.Fatal("unsupported type was accepted")
	}
}

func TestQuantizeModelCreatesOutputWithoutOverwrite(t *testing.T) {
	directory := t.TempDir()
	inputPath := filepath.Join(directory, "input.gguf")
	outputPath := filepath.Join(directory, "output.gguf")
	values := make([]byte, 32*2*4)
	for index := 0; index < 64; index++ {
		value := float32(math.Sin(float64(index) * 0.13))
		binary.LittleEndian.PutUint32(values[index*4:], math.Float32bits(value))
	}
	input, err := os.Create(inputPath)
	if err != nil {
		t.Fatal(err)
	}
	err = gguf.Write(
		input,
		nil,
		[]gguf.TensorData{{
			Name:  "weight",
			Shape: []uint64{32, 2},
			Type:  gguf.DTypeF32,
			Data:  bytes.NewReader(values),
		}},
		gguf.WriteOptions{},
	)
	if closeErr := input.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}

	report, err := quantizeModel(inputPath, outputPath, dtype.Q4_0, false, "")
	if err != nil {
		t.Fatal(err)
	}
	if report.Converted != 1 || report.Preserved != 0 {
		t.Fatalf("report = %#v", report)
	}
	output, err := gguf.Open(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	tensor, ok := output.Tensor("weight")
	if !ok || tensor.Type != dtype.Q4_0 || tensor.Size != 36 {
		t.Fatalf("output tensor = %#v, present=%v", tensor, ok)
	}

	if _, err := quantizeModel(inputPath, outputPath, dtype.Q4_0, false, ""); err == nil {
		t.Fatal("existing output was overwritten")
	}
	if _, err := quantizeModel(inputPath, inputPath, dtype.Q4_0, false, ""); err == nil {
		t.Fatal("identical input and output were accepted")
	}
}

func TestQuantizeModelUsesImportanceMatrix(t *testing.T) {
	directory := t.TempDir()
	inputPath := filepath.Join(directory, "input.gguf")
	outputPath := filepath.Join(directory, "output.gguf")
	imatrixPath := filepath.Join(directory, "imatrix.gguf")
	const name = "blk.0.attn_q.weight"
	values := make([]float32, 256)
	weights := make([]float32, 256)
	for index := range values {
		values[index] = float32(math.Sin(float64(index)*0.13) * 3)
		weights[index] = 0.5 + float32(index%17)/7
	}
	writeGGUFFixture(t, inputPath, nil, []gguf.TensorData{{
		Name: name, Shape: []uint64{256, 1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(encodeFloat32(values)),
	}})
	sums := make([]float32, len(weights))
	for index := range weights {
		sums[index] = weights[index] * 4
	}
	writeGGUFFixture(t, imatrixPath, []gguf.Metadata{
		{Key: "imatrix.datasets", Value: gguf.Value{Type: gguf.ValueTypeArray, ArrayType: gguf.ValueTypeString, Data: []string{"cli-fixture"}}},
		{Key: "imatrix.chunk_count", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(2)}},
		{Key: "imatrix.chunk_size", Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(512)}},
	}, []gguf.TensorData{
		{Name: name + ".in_sum2", Shape: []uint64{256, 1}, Type: gguf.DTypeF32, Data: bytes.NewReader(encodeFloat32(sums))},
		{Name: name + ".counts", Shape: []uint64{1}, Type: gguf.DTypeF32, Data: bytes.NewReader(encodeFloat32([]float32{4}))},
	})

	if _, err := quantizeModel(inputPath, outputPath, dtype.IQ2XXS, false, ""); err == nil {
		t.Fatal("weighted target accepted without -imatrix")
	}
	if _, err := os.Stat(outputPath); !os.IsNotExist(err) {
		t.Fatalf("failed output remains: %v", err)
	}
	report, err := quantizeModel(inputPath, outputPath, dtype.IQ2XXS, false, imatrixPath)
	if err != nil {
		t.Fatal(err)
	}
	if report.Converted != 1 || report.Preserved != 0 {
		t.Fatalf("report = %#v", report)
	}
	output, err := gguf.Open(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	want, err := quant.QuantizeWeighted(dtype.IQ2XXS, values, weights)
	if err != nil {
		t.Fatal(err)
	}
	tensor, ok := output.Tensor(name)
	if !ok || tensor.Type != dtype.IQ2XXS {
		t.Fatalf("output tensor = %#v, present=%v", tensor, ok)
	}
	got := make([]byte, tensor.Size)
	if err := output.ReadTensorData(tensor, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("weighted tensor mismatch")
	}
	for key, expected := range map[string]any{
		"quantize.imatrix.file":          imatrixPath,
		"quantize.imatrix.dataset":       "cli-fixture",
		"quantize.imatrix.entries_count": int64(1),
		"quantize.imatrix.chunks_count":  int64(2),
	} {
		value, ok := output.MetadataValue(key)
		if !ok || value.Data != expected {
			t.Fatalf("%s = %#v, present=%v", key, value, ok)
		}
	}
}

func writeGGUFFixture(t *testing.T, path string, metadata []gguf.Metadata, tensors []gguf.TensorData) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	err = gguf.Write(file, metadata, tensors, gguf.WriteOptions{})
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
}

func encodeFloat32(values []float32) []byte {
	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return data
}
