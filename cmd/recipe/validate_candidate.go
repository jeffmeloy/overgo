package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// validateCandidate compiles one committed candidate recipe against the
// registered capability catalogs and publishes the typed closure verdict —
// runnable or exact refusals — as durable evidence citing the candidate.
func validateCandidate(arguments []string) error {
	flags := flag.NewFlagSet("recipe validate-candidate", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	recipeFlag := flags.String("recipe", "", "committed candidate recipe artifact ID")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*recipeFlag) == "" || flags.NArg() != 0 {
		return errors.New("usage: recipe validate-candidate -recipe <id> [-repo <store>]")
	}
	candidateID, err := artifact.ParseID(*recipeFlag)
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
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	content, found, err := artifact.ReadContent(ctx, store, candidateID)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("recipe: candidate %s is not committed", candidateID)
	}
	definition, err := recipe.ParseDefinition(content.Data)
	if err != nil {
		return err
	}
	closure := modelrecipe.ValidateCandidateRecipe(definition)
	evidence, err := modelrecipe.PublishCandidateClosure(ctx, store, closure)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(map[string]any{"closure": closure, "evidence": evidence}); err != nil {
		return err
	}
	if !closure.Runnable {
		return errors.New("recipe: candidate is not runnable; refusals recorded")
	}
	return nil
}
