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
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

// replayRouting measures one candidate selection policy against recorded
// routing decisions: identical inputs, both policies, published deltas with
// complete coverage. The committed counterfactual report prints with its
// identity — the admission evidence a routing-policy rollout must cite.
func replayRouting(arguments []string) error {
	flags := flag.NewFlagSet("recipe replay", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	policyFlag := flags.String("policy", "", "registered candidate derivation to measure")
	decisionsFlag := flags.String("decisions", "", "comma-separated recorded routing decision artifact IDs")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if strings.TrimSpace(*policyFlag) == "" || strings.TrimSpace(*decisionsFlag) == "" || flags.NArg() != 0 {
		return errors.New("usage: recipe replay -policy <derivation> -decisions <decision-id,...> [-repo <store>]")
	}
	var decisions []artifact.ID
	for text := range strings.SplitSeq(*decisionsFlag, ",") {
		id, err := artifact.ParseID(strings.TrimSpace(text))
		if err != nil {
			return err
		}
		decisions = append(decisions, id)
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
	report, err := modelrecipe.ReplayRoutingDecisions(ctx, store, *policyFlag, decisions)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(report); err != nil {
		return err
	}
	_, err = os.Stdout.WriteString(report.ID.String() + "\n")
	return err
}
