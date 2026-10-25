//overgo:runtime-inputs caller

// Command expert-profile ranks a mixture-of-experts model's routed experts by the decode runs that route them, over a
// fixed corpus of chat requests (corpus.json: prose, code and answers) decoded greedily, and writes the ranking,
// Strata's STRP version 1, beside the model, where every load of the model that keeps experts in host memory places
// the top-ranked ones on the card.
package main

import (
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"

	"overgo/internal/clioptions"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/sampling"
)

//go:embed corpus.json
var corpus []byte

// answerTokens bounds each corpus request's decode: a short answer's routes, the corpus decoding in about a minute.
const answerTokens = 160

func main() {
	clioptions.Main(run)
}

func run() error {
	modelFlags := modelcli.AddModelFlags(flag.CommandLine, "load GGUF LoRA adapter at scale 1; repeatable")
	out := flag.String("out", "", "profile file to write; empty writes the model's own, beside it")
	maxTokens := flag.Int("max-tokens", answerTokens, "tokens each request decodes at most")
	flag.Parse()
	if flag.NArg() != 1 || *maxTokens < 1 {
		return errors.New("usage: expert-profile [options] <model.gguf>")
	}
	path := flag.Arg(0)
	var requests []string
	if err := json.Unmarshal(corpus, &requests); err != nil {
		return err
	}
	ctx := context.Background()
	runner, err := modelFlags.OpenRunner(ctx, path)
	if err != nil {
		return err
	}
	defer runner.Close()
	var decoded int
	for index, request := range requests {
		prompt, err := runner.FormatChat([]inference.ChatMessage{{Role: inference.ChatRoleUser, Content: request}})
		if err != nil {
			return err
		}
		greedy, err := sampling.New(sampling.Config{Temperature: 0})
		if err != nil {
			return err
		}
		ids, _, err := runner.Generate(ctx, prompt, inference.GenerateOptions{MaxNewTokens: *maxTokens, Sampler: greedy, DeviceGreedy: true})
		if err != nil {
			return fmt.Errorf("corpus request %d: %w", index, err)
		}
		decoded += len(ids)
	}
	profile, err := runner.ExpertProfile(ctx)
	if err != nil {
		return err
	}
	destination := cmp.Or(*out, modelcli.ExpertProfilePath(path))
	if err := profile.Write(destination); err != nil {
		return err
	}
	fmt.Printf("wrote %s: %d pairs ranked from %d requests decoding %d tokens\n", destination, len(profile.Pairs), len(requests), decoded)
	return nil
}
