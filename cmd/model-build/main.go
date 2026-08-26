package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/modelbuilder"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
	"overgo/internal/workflowruntime"
)

func main() { clioptions.Main(func() error { return run(os.Args[1:], os.Stdout) }) }

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("model-build", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "overgodb-store", "OvergoDB store")
	dataset := flags.String("dataset", "", "JSON string-array corpus")
	seed := flags.Int64("seed", 1, "construction seed")
	steps := flags.Int("steps", 1, "training steps")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *dataset == "" || flags.NArg() != 0 {
		return errors.New("usage: model-build -dataset corpus.json [-repo path] [-seed n] [-steps n]")
	}
	raw, err := os.ReadFile(*dataset)
	if err != nil {
		return fmt.Errorf("read corpus: %w", err)
	}
	var documents []string
	if err = strictjson.Decode(bytes.NewReader(raw), &documents); err != nil {
		return fmt.Errorf("decode corpus: %w", err)
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	session, err := modelbuilder.NewScratchSession(ctx, modelbuilder.ScratchRequest{
		Repository: store, Documents: documents, Seed: *seed, Steps: *steps,
	})
	if err != nil {
		return err
	}
	operation, err := artifact.JSONID(artifact.KindEvidence, struct {
		Documents []string `json:"documents"`
		Seed      int64    `json:"seed"`
		Steps     int      `json:"steps"`
	}{Documents: documents, Seed: *seed, Steps: *steps})
	if err != nil {
		return err
	}
	state, err := workflowruntime.ExecuteModelBuild(ctx, store, operation, session)
	if err != nil {
		return err
	}
	return json.NewEncoder(output).Encode(state)
}
