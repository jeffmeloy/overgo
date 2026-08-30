package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// declareRolloutPlan commits one immutable rollout plan from a JSON
// specification. The plan is promotion policy, not a feature flag: every
// bound must cite committed evidence, and the store refuses a plan whose
// cited authorities are absent.
func declareRolloutPlan(arguments []string) error {
	flags := flag.NewFlagSet("recipe rollout-plan", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	input := flags.String("input", "", "rollout plan specification JSON path")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*input) == "" || flags.NArg() != 0 {
		return errors.New("usage: recipe rollout-plan -input <rollout.json> [-repo <store>]")
	}
	var specification runrecord.RolloutPlan
	if err := jsonfile.DecodeStrict(*input, &specification); err != nil {
		return err
	}
	plan, err := runrecord.NewRolloutPlan(specification)
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
	batch, err := plan.Batch("rollout/plan/" + plan.ID.String())
	if err != nil {
		return err
	}
	// Prove the exact bytes about to become durable decode back to the same
	// identity before anything is committed.
	decoded, err := runrecord.ParseRolloutPlan(batch.Contents[0].Data)
	if err != nil || decoded.ID != plan.ID {
		return errors.Join(errors.New("recipe: rollout plan content does not reproduce its identity"), err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return err
	}
	fmt.Printf("declared rollout plan %s\n  baseline  %s\n  candidate %s\n  cohort    %d/%d over %q\n",
		plan.ID, plan.Baseline, plan.Candidate,
		plan.CohortShare, runrecord.RolloutCohortDenominator, plan.CohortKey)
	return nil
}
