package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// selectServing derives which recipe serves a task from published
// evaluation and resource evidence in the store: candidates admit through
// the verified evidence gate, the baseline's measured value is the
// threshold, and the recorded decision prints with its committed identity.
func selectServing(arguments []string) error {
	flags := flag.NewFlagSet("recipe select", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	taskFlag := flags.String("task", string(recipe.TaskInference), "recipe task the selection serves")
	metricFlag := flags.String("metric", "", "published evaluation metric the selection is judged on")
	baselineFlag := flags.String("baseline", "", "baseline evaluation evidence artifact ID; its measured value is the threshold")
	candidatesFlag := flags.String("candidates", "", "comma-separated candidate evaluation evidence artifact IDs")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*metricFlag) == "" || strings.TrimSpace(*baselineFlag) == "" ||
		strings.TrimSpace(*candidatesFlag) == "" || flags.NArg() != 0 {
		return errors.New(
			"usage: recipe select -metric <name> -baseline <evidence-id> -candidates <evidence-id,...> [-task <task>] [-repo <store>]",
		)
	}
	baseline, err := artifact.ParseID(*baselineFlag)
	if err != nil {
		return err
	}
	var candidates []artifact.ID
	for _, text := range strings.Split(*candidatesFlag, ",") {
		id, err := artifact.ParseID(strings.TrimSpace(text))
		if err != nil {
			return err
		}
		candidates = append(candidates, id)
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
	decision, err := evaluation.SelectServingRecipe(
		ctx, store, recipe.Task(*taskFlag), *metricFlag, baseline, candidates,
	)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(decision); err != nil {
		return err
	}
	_, err = os.Stdout.WriteString(decision.ID.String() + "\n")
	return err
}
