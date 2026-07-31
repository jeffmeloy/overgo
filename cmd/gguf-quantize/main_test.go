package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

func TestParseQuantizationType(t *testing.T) {
	for input, want := range map[string]dtype.Type{
		"Q4_0":   dtype.Q4_0,
		"q5-1":   dtype.Q5_1,
		" bf16":  dtype.BF16,
		"MXFP4":  dtype.MXFP4,
		"q6-k":   dtype.Q6K,
		"TQ1-0":  dtype.TQ1_0,
		"NVFP4":  dtype.NVFP4,
		"iq2-s":  dtype.IQ2S,
		"iq3_s":  dtype.IQ3S,
		"iq4-xs": dtype.IQ4XS,
	} {
		got, err := parseQuantizationType(input)
		if err != nil || got != want {
			t.Fatalf("parseQuantizationType(%q) = %s, %v; want %s",
				input, got, err, want)
		}
	}
	if _, err := parseQuantizationType("iq2_xxs"); err == nil {
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

	report, err := quantizeModel(inputPath, outputPath, dtype.Q4_0, false)
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

	if _, err := quantizeModel(inputPath, outputPath, dtype.Q4_0, false); err == nil {
		t.Fatal("existing output was overwritten")
	}
	if _, err := quantizeModel(inputPath, inputPath, dtype.Q4_0, false); err == nil {
		t.Fatal("identical input and output were accepted")
	}
}
