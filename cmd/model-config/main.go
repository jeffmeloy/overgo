// model-config records a model's provided configuration in the store:
// the generation essentials and recommended sampling a checkpoint
// directory declares (generation_config.json, config.json) or a model
// card declaration file (model_card_sampling.json with its source URL)
// written for a GGUF-only model. The declaration binds to the GGUF's
// identity, so every runtime and evaluation reads the same record.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

func main() {
	clioptions.MainNamed("model-config", func() error { return run(os.Args[1:]) })
}

func run(args []string) error {
	flags := flag.NewFlagSet("model-config", flag.ContinueOnError)
	repository := flags.String("repo", "", "OvergoDB root; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	source := flags.String("source", "", "checkpoint or declaration directory (generation_config.json, config.json, dna_config.json, model_card_sampling.json)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 || strings.TrimSpace(*source) == "" {
		return errors.New("usage: model-config -source <directory> [-repo <store>] <model.gguf>")
	}
	if strings.TrimSpace(*repository) == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		*repository = roots.Store
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	// The store's identity for the location, not a fresh file digest: a
	// model catalogued by component manifest keeps its manifest identity.
	entries, err := discovery.ServableWithMemo(ctx, store, math.MaxInt, discovery.LoadMemo(ctx, store))
	if err != nil {
		return err
	}
	wanted := filepath.Clean(flags.Arg(0))
	var model artifact.ID
	for _, entry := range entries {
		if strings.EqualFold(filepath.Clean(entry.Location), wanted) {
			model = entry.Model
			break
		}
	}
	if !model.Valid() {
		return fmt.Errorf("model-config: %q is not a servable model location", flags.Arg(0))
	}
	id, recorded, err := modelartifact.RecordModelConfig(ctx, store, *source, model)
	if err != nil {
		return err
	}
	if !recorded {
		fmt.Println("model config: source directory declares no extractable components; nothing committed")
		return nil
	}
	fmt.Printf("model config %s recorded for %s\n", id, flags.Arg(0))
	return nil
}
