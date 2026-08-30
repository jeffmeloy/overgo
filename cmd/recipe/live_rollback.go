package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/loop"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// liveRollback rolls the active alias back after a containment-level
// breaker transition, binding the exact judged candidate so a stale
// decision refuses once the alias has moved, and printing the published
// causal record identity.
func liveRollback(arguments []string) error {
	flags := flag.NewFlagSet("recipe live-rollback", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	modelFlag := flags.String("model", "", "model artifact ID whose task activation rolls back")
	taskFlag := flags.String("task", string(recipe.TaskInference), "recipe task")
	judgedFlag := flags.String("judged", "", "recipe artifact ID the breaker judged; must still be active")
	transitionFlag := flags.String("transition", "", "published containment-level breaker transition artifact ID")
	failedGate := flags.String("failed-gate", "", "failed verifier gate artifact ID for the judged recipe")
	failedRun := flags.String("failed-run", "", "failed verifier run artifact ID for the judged recipe")
	gateFlag := flags.String("gate", "", "successful verifier gate artifact ID for the predecessor")
	runFlag := flags.String("run-id", "", "bound verifier run artifact ID for the predecessor")
	reason := flags.String("reason", "", "rollback reason recorded on every transition")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*modelFlag) == "" || strings.TrimSpace(*judgedFlag) == "" ||
		strings.TrimSpace(*transitionFlag) == "" || strings.TrimSpace(*reason) == "" || flags.NArg() != 0 {
		return errors.New(
			"usage: recipe live-rollback -model <id> -judged <recipe-id> -transition <id> -failed-gate <id> -failed-run <id> -gate <id> -run-id <id> -reason <text> [-task <task>] [-repo <store>]",
		)
	}
	modelID, err := artifact.ParseID(strings.TrimSpace(*modelFlag))
	if err != nil {
		return err
	}
	judged, err := artifact.ParseID(strings.TrimSpace(*judgedFlag))
	if err != nil {
		return err
	}
	transition, err := artifact.ParseID(strings.TrimSpace(*transitionFlag))
	if err != nil {
		return err
	}
	retirement, err := parseVerification(*failedGate, *failedRun)
	if err != nil {
		return err
	}
	reactivation, err := parseVerification(*gateFlag, *runFlag)
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
	record, err := loop.RollbackQuarantinedCandidate(
		ctx, store, modelID, recipe.Task(*taskFlag), judged, transition,
		retirement, reactivation, *reason,
	)
	if err != nil {
		return err
	}
	fmt.Printf("rolled back %s %s\n  causal record %s\n", *taskFlag, modelID, record)
	return nil
}
