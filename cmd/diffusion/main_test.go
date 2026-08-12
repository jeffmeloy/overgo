package main

import (
	"testing"

	"overgo/internal/inference"
)

func TestParseCLI(t *testing.T) {
	config, err := parseCLI([]string{
		"-length", "64", "-steps", "16", "-algorithm", "2",
		"-block-length", "8", "-cfg-scale", "1.5", "-alg-temp", "0.2",
		"-temp", "0.4", "-top-k", "12", "-top-p", "0.8", "-seed", "7",
		"-gumbel", "-shift-logits", "false",
		"-lora", "a.gguf", "model.gguf", "prompt",
	})
	if err != nil {
		t.Fatal(err)
	}
	options := config.diffusion
	if config.model != "model.gguf" || config.prompt != "prompt" ||
		len(config.lora) != 1 || options.MaxLength != 64 || options.Steps != 16 ||
		options.Algorithm != inference.DiffusionMargin || options.Schedule != inference.DiffusionBlock ||
		options.BlockLength != 8 || options.CFGScale != 1.5 || options.AlgorithmTemperature != 0.2 ||
		options.Temperature != 0.4 || options.TopK != 12 || options.TopP != 0.8 || options.Seed != 7 ||
		!options.AddGumbelNoise || options.ShiftLogits == nil || *options.ShiftLogits {
		t.Fatalf("config = %+v", config)
	}
}

func TestParseCLIRejectsInvalidFlags(t *testing.T) {
	for _, arguments := range [][]string{
		{"model.gguf"},
		{"model.gguf", "prompt"},
		{"-eps", "0.001", "-shift-logits", "sometimes", "model.gguf", "prompt"},
		{"-eps", "0.001", "-preload", "-native-quant", "model.gguf", "prompt"},
		{"-eps", "0.001", "-lora", "", "model.gguf", "prompt"},
		{"-eps", "0.001", "-block-length", "8", "model.gguf", "prompt"},
		{"-block-length", "7", "model.gguf", "prompt"},
		{"-eps", "0.001", "-algorithm", "5", "model.gguf", "prompt"},
	} {
		if _, err := parseCLI(arguments); err == nil {
			t.Fatalf("arguments accepted: %v", arguments)
		}
	}
}
