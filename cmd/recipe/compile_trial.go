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
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/trainingprogram"
)

// compilePrototypeTrial resolves a declared prototype from the store and
// compiles its closed-world trial plan for inspection: everything derives
// through registered Go capabilities and nothing executes.
func compilePrototypeTrial(arguments []string) error {
	flags := flag.NewFlagSet("recipe compile-trial", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	prototypeFlag := flags.String("prototype", "", "declared model prototype artifact ID")
	input := flags.String("input", "", "trial specification JSON path")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*prototypeFlag) == "" || strings.TrimSpace(*input) == "" || flags.NArg() != 0 {
		return errors.New("usage: recipe compile-trial -prototype <id> -input <trial.json> [-repo <store>]")
	}
	prototypeID, err := artifact.ParseID(*prototypeFlag)
	if err != nil {
		return err
	}
	var specification struct {
		Admission      artifact.ID                   `json:"admission"`
		Construction   trainingprogram.ScratchSpec   `json:"construction"`
		Objective      trainingprogram.ObjectiveKind `json:"objective"`
		EvaluationPlan artifact.ID                   `json:"evaluation_plan"`
		Heads          uint32                        `json:"heads"`
	}
	if err := jsonfile.DecodeStrict(*input, &specification); err != nil {
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
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	content, found, err := artifact.ReadContent(context.Background(), store, prototypeID)
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
	trial, err := modelrecipe.CompilePrototypeTrial(
		prototype, specification.Admission, specification.Construction,
		specification.Objective, specification.EvaluationPlan, specification.Heads,
	)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(map[string]any{
		"prototype": trial.Prototype.ID, "admission": trial.Admission,
		"architecture": trial.Architecture.Name, "draft_heads": trial.Draft.Heads,
		"construction": trial.Construction.ID(), "objective": trial.Objective,
		"evaluation_plan": trial.EvaluationPlan, "candidate_recipes": trial.CandidateRecipes,
	})
}
