package main

import (
	"context"
	"errors"
	"flag"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/inference"
)

func main() {
	clioptions.Main(run)
}

func run() error {
	modelFlags := clioptions.AddModelFlags(flag.CommandLine, "load GGUF LoRA adapter at scale 1; repeatable")
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
	runner, err := modelFlags.OpenRunner(context.Background(), flag.Arg(0))
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
	return clioptions.WritePrettyJSON(os.Stdout, result)
}
