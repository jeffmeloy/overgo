package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"llamacpp2go/internal/inference"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	deviceOrdinal := flag.Int("device", 0, "CUDA device ordinal")
	preload := flag.Bool("preload", false, "dequantize all model weights once into CUDA memory")
	nativeQ8 := flag.Bool("native-q8", false, "preload Q8_0 weights without dequantizing them")
	nativeQuant := flag.Bool("native-quant", false, "preload supported quantized weights without dequantizing them")
	textFile := flag.String("file", "", "read evaluation text from this UTF-8 file")
	includeScores := flag.Bool("token-scores", false, "include per-token negative log-likelihoods")
	contextSize := flag.Int(
		"ctx",
		0,
		"llama.cpp-compatible disjoint scoring window; zero scores every token",
	)
	flag.Parse()
	if flag.NArg() < 1 || flag.NArg() > 2 || (*textFile != "" && flag.NArg() != 1) {
		return errors.New("usage: perplexity [options] <model.gguf> [text]")
	}
	var text string
	if *textFile != "" {
		data, err := os.ReadFile(*textFile)
		if err != nil {
			return err
		}
		text = string(data)
	} else {
		if flag.NArg() != 2 {
			return errors.New("perplexity text or -file is required")
		}
		text = flag.Arg(1)
	}
	runner, err := inference.OpenWithOptions(flag.Arg(0), inference.OpenOptions{
		DeviceOrdinal:           *deviceOrdinal,
		PreloadDeviceWeights:    *preload,
		PreloadQuantizedWeights: *nativeQ8 || *nativeQuant,
	})
	if err != nil {
		return err
	}
	defer runner.Close()
	result, err := runner.PerplexityWithOptions(
		context.Background(),
		text,
		inference.PerplexityOptions{ContextSize: *contextSize},
	)
	if err != nil {
		return err
	}
	if !*includeScores {
		result.Scores = nil
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(result)
}
