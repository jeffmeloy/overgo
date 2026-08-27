// Command automation-policy drives the evidence-gated lifecycle of
// automation policies from the operator's CLI: declare a challenger,
// promote it one stage at a time on recorded evidence, inspect a
// slot, and roll an activation back to its retained incumbent.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/automationpolicy"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
)

func main() {
	clioptions.MainNamed("automation-policy", run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "OvergoDB store directory")
	strategy := flag.String("strategy", "", "declare: strategy identity the candidate proposes")
	evidence := flag.String("evidence", "", "promote: comma-separated recorded evidence artifact IDs")
	flag.Parse()
	if flag.NArg() != 2 {
		return errors.New("usage: automation-policy <declare|promote|rollback|status> [-strategy s] [-evidence ids] [-repo store] <slot>")
	}
	verb, slot := flag.Arg(0), flag.Arg(1)
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	switch verb {
	case "declare":
		if *strategy == "" {
			return errors.New("automation-policy: declare requires -strategy")
		}
		policy, err := automationpolicy.Declare(ctx, store, slot, *strategy)
		if err != nil {
			return err
		}
		return print(policy)
	case "promote":
		ids, err := parseEvidence(*evidence)
		if err != nil {
			return err
		}
		policy, err := automationpolicy.Promote(ctx, store, slot, ids)
		if err != nil {
			return err
		}
		return print(policy)
	case "rollback":
		policy, err := automationpolicy.Rollback(ctx, store, slot)
		if err != nil {
			return err
		}
		return print(policy)
	case "status":
		governing, standing, err := automationpolicy.Load(ctx, store, slot)
		if err != nil {
			return err
		}
		if standing {
			fmt.Print("governing: ")
			if err := print(governing); err != nil {
				return err
			}
		} else {
			fmt.Println("governing: none")
		}
		candidate, mid, err := automationpolicy.LoadCandidate(ctx, store, slot)
		if err != nil {
			return err
		}
		if mid && (!standing || candidate.ID != governing.ID) {
			fmt.Print("candidate: ")
			return print(candidate)
		}
		return nil
	default:
		return fmt.Errorf("automation-policy: unknown verb %q", verb)
	}
}

func parseEvidence(csv string) ([]artifact.ID, error) {
	if strings.TrimSpace(csv) == "" {
		return nil, nil
	}
	var ids []artifact.ID
	for _, raw := range strings.Split(csv, ",") {
		id, err := artifact.ParseID(strings.TrimSpace(raw))
		if err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}

func print(policy automationpolicy.Policy) error {
	encoded, err := json.MarshalIndent(struct {
		automationpolicy.Policy
		ID artifact.ID `json:"id"`
	}{Policy: policy, ID: policy.ID}, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}
