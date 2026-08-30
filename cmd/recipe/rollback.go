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
)

// rollbackActivation returns a model task to its previous recipe after the
// active one was retired with failed proof: the predecessor re-earns active
// through the supplied fresh gate/run verification bound to its identity.
func rollbackActivation(arguments []string) error {
	flags := flag.NewFlagSet("recipe rollback", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	modelFlag := flags.String("model", "", "model artifact ID whose task activation rolls back")
	taskFlag := flags.String("task", string(recipe.TaskInference), "recipe task")
	gateFlag := flags.String("gate", "", "successful verifier gate artifact ID for the predecessor")
	runFlag := flags.String("run-id", "", "bound verifier run artifact ID for the predecessor")
	reason := flags.String("reason", "", "rollback reason recorded in the decision event")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*modelFlag) == "" || strings.TrimSpace(*reason) == "" || flags.NArg() != 0 {
		return errors.New(
			"usage: recipe rollback -model <id> -gate <id> -run-id <id> -reason <text> [-task <task>] [-repo <store>]",
		)
	}
	modelID, err := artifact.ParseID(*modelFlag)
	if err != nil {
		return err
	}
	verification, err := parseVerification(*gateFlag, *runFlag)
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
	if err := modelrecipe.RollbackActivation(
		ctx, store, modelID, recipe.Task(*taskFlag), verification, recipe.EvidenceVerified, *reason,
	); err != nil {
		return err
	}
	fmt.Printf("rolled back %s %s\n  reason %s\n", *taskFlag, modelID, *reason)
	return nil
}
