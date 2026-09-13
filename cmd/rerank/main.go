package main

import (
	"context"
	"errors"
	"flag"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/modelcli"
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
	modelFlags := modelcli.AddModelFlags(flags, "load GGUF LoRA adapter at scale 1; repeatable")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if *modelPath == "" {
		return errors.New("-model is required")
	}
	if *query == "" || *document == "" {
		return errors.New("-query and -document are required")
	}
	runner, err := modelFlags.OpenRunner(context.Background(), *modelPath)
	if err != nil {
		return err
	}
	defer runner.Close()
	result, err := runner.RankPair(context.Background(), *query, *document)
	if err != nil {
		return err
	}
	return clioptions.WritePrettyJSON(os.Stdout, report{
		Model: runner.Spec().Name, Tokens: result.Tokens,
		Labels: result.Labels, Scores: result.Scores,
	})
}

func main() {
	clioptions.Main(run)
}
