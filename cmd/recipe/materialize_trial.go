package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// materializeTrial publishes one successful prototype trial atomically: the
// trained model file's derived inventory and resolved definition bound to
// the prototype's exact architecture and profile, plus the candidate recipe
// citations — one store commit, no alias mutation.
func materializeTrial(arguments []string) error {
	flags := flag.NewFlagSet("recipe materialize-trial", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	prototypeFlag := flags.String("prototype", "", "declared model prototype artifact ID")
	admissionFlag := flags.String("admission", "", "improvement admission evidence ID")
	runFlag := flags.String("run", "", "trial run receipt artifact ID")
	candidatesFlag := flags.String("candidates", "", "comma-separated candidate recipe IDs")
	tasksFlag := flags.String("tasks", "", "comma-separated tasks whose candidate recipes derive from the resolved definition")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if flags.NArg() != 1 || strings.TrimSpace(*prototypeFlag) == "" || strings.TrimSpace(*runFlag) == "" ||
		strings.TrimSpace(*admissionFlag) == "" ||
		strings.TrimSpace(*candidatesFlag) == "" == (strings.TrimSpace(*tasksFlag) == "") {
		return errors.New("usage: recipe materialize-trial -prototype <id> -admission <id> -run <id> (-candidates <csv> | -tasks <csv>) [-repo <store>] <model>")
	}
	prototypeID, err := artifact.ParseID(*prototypeFlag)
	if err != nil {
		return err
	}
	admission, err := artifact.ParseID(*admissionFlag)
	if err != nil {
		return err
	}
	trialRun, err := artifact.ParseID(*runFlag)
	if err != nil {
		return err
	}
	candidates := []artifact.ID{}
	if strings.TrimSpace(*candidatesFlag) != "" {
		for _, raw := range strings.Split(*candidatesFlag, ",") {
			candidate, parseErr := artifact.ParseID(strings.TrimSpace(raw))
			if parseErr != nil {
				return parseErr
			}
			candidates = append(candidates, candidate)
		}
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	repository := strings.TrimSpace(*repoFlag)
	if repository == "" {
		repository = roots.Store
	}
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	content, found, err := artifact.ReadContent(ctx, store, prototypeID)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("recipe: prototype %s is not committed", prototypeID)
	}
	prototype, err := modelrecipe.ParseModelPrototype(content.Data)
	if err != nil {
		return err
	}
	candidate, err := prepareInferenceCandidate(
		ctx, store, roots.ResolveModelPath(flags.Arg(0)), sessionOverride{}, recipe.ResidencyHybridNative,
	)
	if err != nil {
		return err
	}
	trial := modelrecipe.PrototypeTrial{
		Prototype: prototype, Admission: admission,
		Objective: trainingprogram.ObjectiveTokenPrediction,
	}
	derivedContents := []artifact.Content{}
	if strings.TrimSpace(*tasksFlag) != "" {
		tasks := []recipe.Task{}
		for _, raw := range strings.Split(*tasksFlag, ",") {
			tasks = append(tasks, recipe.Task(strings.TrimSpace(raw)))
		}
		derived, deriveErr := modelrecipe.DerivePrototypeRecipes(
			candidate.resolved, recipe.PlacementHybrid, recipe.SessionCapacity,
			recipe.ResidencyHybridNative, tasks,
		)
		if deriveErr != nil {
			return deriveErr
		}
		for _, definition := range derived {
			definitionContent, contentErr := modelrecipe.Content(definition)
			if contentErr != nil {
				return contentErr
			}
			derivedContents = append(derivedContents, definitionContent)
			candidates = append(candidates, definition.ID)
		}
	}
	batch, err := modelrecipe.MaterializePrototypeTrial(
		trial, candidate.resolved, candidate.inventory, trialRun, candidates,
	)
	if err != nil {
		return err
	}
	batch.Contents = append(batch.Contents, derivedContents...)
	if err := batch.Validate(); err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Printf("materialized prototype trial %s: definition %s, %d candidate recipe(s)\n",
		prototype.ID, candidate.resolved.Document.ID, len(candidates))
	return nil
}
