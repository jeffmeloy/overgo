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
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
)

// projectRollout materializes one immutable rollout reading at the store's
// current head — head, projector version, query contract, sources,
// coverage, digest — publishing it as the evidence promotion and rollback
// consume; with -parse, a committed report re-parses and proves that its
// digest reproduces from its own bindings.
func projectRollout(arguments []string) error {
	flags := flag.NewFlagSet("recipe rollout-project", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	planFlag := flags.String("plan", "", "committed rollout plan artifact ID to project")
	parseFlag := flags.String("parse", "", "committed projection report artifact ID to re-parse and audit")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	hasPlan, hasParse := strings.TrimSpace(*planFlag) != "", strings.TrimSpace(*parseFlag) != ""
	if hasPlan == hasParse || flags.NArg() != 0 {
		return errors.New("usage: recipe rollout-project -plan <plan-id> | -parse <report-id> [-repo <store>]")
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
	var projection evaluation.RolloutProjection
	if hasParse {
		reportID, err := artifact.ParseID(*parseFlag)
		if err != nil {
			return err
		}
		content, found, err := artifact.ReadContent(ctx, store, reportID)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("recipe: projection report %s is not committed", reportID)
		}
		projection, err = evaluation.ParseRolloutProjection(content.Data)
		if err != nil {
			return err
		}
	} else {
		planID, err := artifact.ParseID(*planFlag)
		if err != nil {
			return err
		}
		projection, err = evaluation.ProjectRolloutEvidence(ctx, store, planID)
		if err != nil {
			return err
		}
		if _, err := evaluation.PublishRolloutProjection(ctx, store, projection); err != nil {
			return err
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(projection); err != nil {
		return err
	}
	_, err = os.Stdout.WriteString(projection.ID.String() + "\n")
	return err
}
