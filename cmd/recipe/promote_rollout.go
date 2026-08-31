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
)

// promoteRollout activates a rollout's candidate after the required
// evidence closure: plan, baseline reading, boundary reading, observation
// coverage, rollback contract, and the plan's baseline in live service.
func promoteRollout(arguments []string) error {
	flags := flag.NewFlagSet("recipe promote-rollout", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	planFlag := flags.String("plan", "", "committed rollout plan artifact ID")
	baselineFlag := flags.String("baseline-reading", "", "projection reading at the rollout's baseline head")
	boundaryFlag := flags.String("boundary-reading", "", "projection reading at the promotion boundary head")
	gateFlag := flags.String("gate", "", "successful verifier gate artifact ID for the candidate")
	runFlag := flags.String("run-id", "", "bound verifier run artifact ID for the candidate")
	reason := flags.String("reason", "", "promotion reason recorded in the decision event")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*planFlag) == "" || strings.TrimSpace(*baselineFlag) == "" ||
		strings.TrimSpace(*boundaryFlag) == "" || strings.TrimSpace(*reason) == "" || flags.NArg() != 0 {
		return errors.New(
			"usage: recipe promote-rollout -plan <id> -baseline-reading <id> -boundary-reading <id> -gate <id> -run-id <id> -reason <text> [-repo <store>]",
		)
	}
	closure := loop.RolloutPromotionClosure{}
	for _, binding := range []struct {
		text   string
		target *artifact.ID
	}{
		{*planFlag, &closure.Plan},
		{*baselineFlag, &closure.BaselineReading},
		{*boundaryFlag, &closure.CandidateReading},
	} {
		id, err := artifact.ParseID(strings.TrimSpace(binding.text))
		if err != nil {
			return err
		}
		*binding.target = id
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
	if err := loop.PromoteRolloutCandidate(ctx, store, closure, verification, *reason); err != nil {
		return err
	}
	fmt.Printf("promoted rollout %s\n  reason %s\n", closure.Plan, *reason)
	return nil
}
