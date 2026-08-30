package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

// declarePrototype commits one immutable model-prototype hypothesis from a
// JSON specification. Declaring authorizes nothing: the document only
// becomes constructible work through admission, and executable only through
// the compiled trial the later lifecycle rows own.
func declarePrototype(arguments []string) error {
	flags := flag.NewFlagSet("recipe declare-prototype", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	input := flags.String("input", "", "model prototype specification JSON path")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*input) == "" || flags.NArg() != 0 {
		return errors.New("usage: recipe declare-prototype -input <prototype.json> [-repo <store>]")
	}
	var specification modelrecipe.ModelPrototype
	if err := jsonfile.DecodeStrict(*input, &specification); err != nil {
		return err
	}
	prototype, err := modelrecipe.NewModelPrototype(specification)
	if err != nil {
		return err
	}
	repository := strings.TrimSpace(*repoFlag)
	if repository == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		repository = roots.Store
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	batch, err := prototype.Batch("model-prototype/" + prototype.ID.String())
	if err != nil {
		return err
	}
	// Prove the exact bytes about to become durable decode back to the same
	// identity before anything is committed.
	decoded, err := modelrecipe.ParseModelPrototype(batch.Contents[0].Data)
	if err != nil || decoded.ID != prototype.ID {
		return errors.Join(errors.New("recipe: prototype content does not reproduce its identity"), err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return err
	}
	fmt.Printf("declared model prototype %s (architecture %s)\n", prototype.ID, prototype.Architecture)
	return nil
}
