package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/inference"
	"overgo/internal/model"
	"overgo/internal/tokenizer"
)

type report struct {
	TokenIDs   []tokenizer.TokenID `json:"token_ids"`
	Tokens     int                 `json:"tokens"`
	Dimensions int                 `json:"dimensions"`
	Pooling    string              `json:"pooling"`
	Embedding  []float32           `json:"embedding"`
	L2Norm     float64             `json:"l2_norm"`
}

func run() error {
	flags := flag.NewFlagSet("embedding", flag.ContinueOnError)
	modelPath := flags.String("model", "", "GGUF encoder model path")
	prompt := flags.String("prompt", "", "text to encode")
	modelFlags := clioptions.AddModelFlagsWithConfig(flags, "load GGUF LoRA adapter at scale 1; repeatable", clioptions.ModelFlagConfig{
		PreloadName: "preload-f32", NativeQuantName: "native-quant", NativeQuantDefault: true,
	})
	pooling := flags.String("pooling", "mean", "pooling: mean, last, or none")
	normalize := flags.Int("normalize", -1, "embedding normalization: -1 none, 0 max-absolute, or p-norm")
	dimensions := flags.Int("dimensions", 0, "limit emitted embedding dimensions (0 = all)")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if *modelPath == "" {
		return errors.New("-model is required")
	}
	runner, err := inference.OpenWithOptions(*modelPath, modelFlags.OpenOptions(1))
	if err != nil {
		return err
	}
	defer runner.Close()
	if runner.Spec().Profile().Forward != model.ForwardT5Encoder {
		return fmt.Errorf("embedding: architecture %q is not an encoder", runner.Spec().Architecture)
	}
	ids, err := runner.Vocab().Encode(*prompt, tokenizer.EncodeOptions{AddSpecial: true})
	if err != nil {
		return err
	}
	result, err := runner.EmbedTokensAdvanced(
		context.Background(),
		ids,
		inference.EmbeddingOptions{
			Pooling:   inference.EmbeddingPooling(*pooling),
			Normalize: *normalize,
		},
	)
	if err != nil {
		return err
	}
	width := 0
	if len(result.Vectors) > 0 {
		width = len(result.Vectors[0])
	}
	embedding := make([]float32, 0, width*len(result.Vectors))
	for _, vector := range result.Vectors {
		embedding = append(embedding, vector...)
	}
	var squared float64
	for _, value := range embedding {
		squared += float64(value) * float64(value)
	}
	emitted := embedding
	if *dimensions < 0 {
		return errors.New("-dimensions cannot be negative")
	}
	if *dimensions > 0 && *dimensions < len(emitted) {
		emitted = emitted[:*dimensions]
	}
	return clioptions.WriteJSON(os.Stdout, report{
		TokenIDs:   ids,
		Tokens:     result.Tokens,
		Dimensions: width,
		Pooling:    *pooling,
		Embedding:  emitted,
		L2Norm:     math.Sqrt(squared),
	})
}

func main() {
	clioptions.Main(run)
}
