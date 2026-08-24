// Command lora-extract derives and records an exact-base LoRA adapter.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/modelmerge"
	"overgo/internal/overgodb"
)

type extractionRequest struct {
	Base   extractionSnapshot              `json:"base"`
	Tuned  extractionSnapshot              `json:"tuned"`
	Policy modelmerge.LoRAExtractionPolicy `json:"policy"`
}

type extractionSnapshot struct {
	ID         artifact.ID                  `json:"id"`
	Definition artifact.ID                  `json:"definition"`
	Base       artifact.ID                  `json:"base,omitzero"`
	Tensors    map[string]modelmerge.Weight `json:"tensors"`
}

func main() {
	clioptions.MainNamed("lora-extract", run)
}

func run() error {
	flags := flag.NewFlagSet("lora-extract", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	requestPath := flags.String("request", "", "exact snapshots and extraction policy JSON")
	repository := flags.String("record", "", "OvergoDB root")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if flags.NArg() != 0 || *requestPath == "" || *repository == "" {
		return errors.New("usage: lora-extract -request JSON -record STORE")
	}
	var request extractionRequest
	if err := jsonfile.Decode(*requestPath, &request); err != nil {
		return err
	}
	base := request.Base.snapshot()
	tuned := request.Tuned.snapshot()
	extraction, err := modelmerge.ExtractLoRA(base, tuned, request.Policy)
	if err != nil {
		return err
	}
	if _, err := extraction.Reconstruct(base); err != nil {
		return err
	}
	content, err := extraction.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		"lora-extraction/"+extraction.ID.String(), []artifact.Content{content}, extraction.Lineage(), nil,
	)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return err
	}
	fmt.Printf("LORA EXTRACTION GREEN adapter=%s base=%s tuned=%s tensors=%d\n",
		extraction.ID, extraction.Base, extraction.Tuned, len(extraction.Weights))
	return nil
}

func (value extractionSnapshot) snapshot() modelmerge.Snapshot {
	return modelmerge.Snapshot{
		ID: value.ID, Definition: value.Definition, Base: value.Base, Tensors: value.Tensors,
	}
}
