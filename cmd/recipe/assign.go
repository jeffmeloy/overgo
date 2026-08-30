package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// assignRollout resolves which arm serves one workload unit under a
// committed rollout plan. The assignment is the registered deterministic
// scheme, so any process asking for the same unit prints the same arm.
func assignRollout(arguments []string) error {
	flags := flag.NewFlagSet("recipe assign", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	planFlag := flags.String("plan", "", "committed rollout plan artifact ID")
	unitFlag := flags.String("unit", "", "workload unit within the plan's cohort key domain")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*planFlag) == "" || strings.TrimSpace(*unitFlag) == "" || flags.NArg() != 0 {
		return errors.New("usage: recipe assign -plan <plan-id> -unit <unit> [-repo <store>]")
	}
	planID, err := artifact.ParseID(*planFlag)
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
	content, found, err := artifact.ReadContent(ctx, store, planID)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("recipe: rollout plan %s is not committed", planID)
	}
	plan, err := runrecord.ParseRolloutPlan(content.Data)
	if err != nil {
		return err
	}
	arm, err := plan.Assign(*unitFlag)
	if err != nil {
		return err
	}
	role := "baseline"
	if arm == plan.Candidate {
		role = "candidate"
	}
	fmt.Printf("%s %s\n", role, arm)
	return nil
}
