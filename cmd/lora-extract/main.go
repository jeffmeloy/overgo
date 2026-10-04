// Command lora-extract derives and records an exact-base LoRA adapter.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/modelmerge"
	"overgo/internal/overgodb"
	"overgo/internal/peft"
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

func run() error { return extract(os.Args[1:]) }

func extract(args []string) error {
	flags := flag.NewFlagSet("lora-extract", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	requestPath := flags.String("request", "", "exact snapshots and extraction policy JSON")
	repository := flags.String("record", "", "OvergoDB root")
	peftDirectory := flags.String("peft", "", "also write the adapter here as a Hugging Face PEFT directory; the snapshots' tensors must be named as their model's .weight tensors")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *requestPath == "" || *repository == "" {
		return errors.New("usage: lora-extract -request JSON -record STORE [-peft DIRECTORY]")
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
	// A name PEFT cannot attach factors to is refused before anything is recorded.
	var adapter peft.Adapter
	if *peftDirectory != "" {
		if adapter, err = peftAdapter(extraction); err != nil {
			return err
		}
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
	if *peftDirectory == "" {
		return nil
	}
	if err := peft.Save(*peftDirectory, adapter); err != nil {
		return err
	}
	fmt.Printf("PEFT adapter written to %s: %d modules, rank up to %d\n", *peftDirectory, len(adapter.Modules), adapter.Config.Rank)
	return nil
}

// peftAdapter is the extraction as a PEFT adapter. Each module's factors
// already carry the whole update, so its alpha is its rank and every scale is
// one; the config's rank is the largest, and the others take pattern entries.
func peftAdapter(extraction modelmerge.LoRAExtraction) (peft.Adapter, error) {
	modules := make(map[string]peft.Module, len(extraction.Weights))
	rank := 0
	for name, factors := range extraction.Weights {
		module, isWeight := strings.CutSuffix(name, ".weight")
		if !isWeight {
			return peft.Adapter{}, fmt.Errorf("lora-extract: tensor %q is not a .weight: a PEFT adapter names the module whose weight it adapts", name)
		}
		modules[module] = peft.Module{
			Rank: int(factors.Rank), In: int(factors.Columns), Out: int(factors.Rows), Alpha: float64(factors.Rank), Down: factors.Down, Up: factors.Up,
		}
		rank = max(rank, int(factors.Rank))
	}
	return peft.Adapter{Config: peft.Config{BaseModel: extraction.Base.String(), Rank: rank, Alpha: float64(rank)}, Modules: modules}, nil
}

func (value extractionSnapshot) snapshot() modelmerge.Snapshot {
	return modelmerge.Snapshot{
		ID: value.ID, Definition: value.Definition, Base: value.Base, Tensors: value.Tensors,
	}
}
