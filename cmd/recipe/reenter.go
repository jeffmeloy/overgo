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

// reenterQuarantine admits one quarantined candidate back into rollout
// through the complete new-evidence closure: prior finding, changed
// mechanism, new evaluation, fresh rollout plan, and an explicit decider.
func reenterQuarantine(arguments []string) error {
	flags := flag.NewFlagSet("recipe reenter", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	candidateFlag := flags.String("candidate", "", "recipe artifact ID re-entering rollout")
	findingFlag := flags.String("finding", "", "published rollback finding artifact ID the reentry answers")
	mechanismFlag := flags.String("mechanism", "", "changed-mechanism evidence artifact ID, newer than the finding")
	evaluationFlag := flags.String("evaluation", "", "new evaluation evidence artifact ID, newer than the finding")
	planFlag := flags.String("plan", "", "fresh rollout plan artifact ID governing the candidate")
	deciderFlag := flags.String("decider", "", "explicit recorded decider")
	reason := flags.String("reason", "", "reentry reason recorded on the admission")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	reentry := loop.QuarantineReentry{Decider: *deciderFlag, Reason: *reason}
	bindings := []struct {
		text   string
		target *artifact.ID
	}{
		{*candidateFlag, &reentry.Candidate},
		{*findingFlag, &reentry.Finding},
		{*mechanismFlag, &reentry.Mechanism},
		{*evaluationFlag, &reentry.Evaluation},
		{*planFlag, &reentry.Plan},
	}
	for _, binding := range bindings {
		if strings.TrimSpace(binding.text) == "" {
			return errors.New(
				"usage: recipe reenter -candidate <id> -finding <id> -mechanism <id> -evaluation <id> -plan <id> -decider <who> -reason <text> [-repo <store>]",
			)
		}
		id, err := artifact.ParseID(strings.TrimSpace(binding.text))
		if err != nil {
			return err
		}
		*binding.target = id
	}
	if flags.NArg() != 0 {
		return errors.New("recipe reenter takes no positional arguments")
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
	admitted, err := loop.AdmitQuarantineReentry(ctx, store, reentry)
	if err != nil {
		return err
	}
	fmt.Printf("admitted reentry %s\n  candidate %s\n  decider   %s\n", admitted, reentry.Candidate, reentry.Decider)
	return nil
}
