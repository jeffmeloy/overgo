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

type report struct {
	Model  string    `json:"model"`
	Tokens int       `json:"tokens"`
	Labels []string  `json:"labels"`
	Scores []float32 `json:"scores"`
}

func run() error {
	flags := flag.NewFlagSet("rerank", flag.ContinueOnError)
	modelPath := flags.String("model", "", "Qwen3 or Qwen3-VL reranker GGUF path")
	query := flags.String("query", "", "query text")
	document := flags.String("document", "", "document text")
	device := flags.Int("device", 0, "CUDA device ordinal")
	nativeQuant := flags.Bool("native-quant", true, "preload quantized weights in native form")
	preloadF32 := flags.Bool("preload-f32", false, "preload all weights as F32")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if *modelPath == "" {
		return errors.New("-model is required")
	}
	if *query == "" || *document == "" {
		return errors.New("-query and -document are required")
	}
	runner, err := inference.OpenWithOptions(*modelPath, inference.OpenOptions{
		DeviceOrdinal:           *device,
		PreloadDeviceWeights:    *preloadF32,
		PreloadQuantizedWeights: *nativeQuant,
	})
	if err != nil {
		return err
	}
	defer runner.Close()
	result, err := runner.RankPair(context.Background(), *query, *document)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report{
		Model: runner.Spec().Name, Tokens: result.Tokens,
		Labels: result.Labels, Scores: result.Scores,
	})
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
