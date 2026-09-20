package main

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// openPlanStore opens the store a plan command reads. An open costs seconds
// (3.7 s measured on the store in service), so a command opens once and
// shares the handle between dispatch and the review.
var openPlanStore = func() (*overgodb.Store, error) {
	return overgodb.OpenReadOnly(filepath.Join(commandWorktree, gitauthority.CanonicalOvergoDBDirectory))
}

// pendingOptimizationCandidates derives the landed completion's candidates
// and keeps those the plan has not answered.
func pendingOptimizationCandidates(document plan.Plan, store *overgodb.Store) ([]plan.OptimizationCandidate, error) {
	candidates, err := plan.ReviewLandedCompletion(context.Background(), store, commandWorktree, "HEAD")
	if err != nil {
		return nil, err
	}
	return plan.UnreviewedCandidates(candidates, document), nil
}

// requireOptimizationReview refuses dispatch while the last landed
// completion's optimization candidates lack a disposition, printing each so
// the re-plan can answer it.
func requireOptimizationReview(document plan.Plan, store *overgodb.Store, printLines bool, output io.Writer) error {
	pending, err := pendingOptimizationCandidates(document, store)
	if err != nil || len(pending) == 0 {
		return err
	}
	if printLines {
		for _, candidate := range pending {
			fmt.Fprintf(output, "review pending: %s %s: %s\n", candidate.Kind, candidate.Key, candidate.Measure)
		}
	}
	return fmt.Errorf("plan: optimization review pending for %d candidate(s) of the landed completion; record each with plan -review <key> or every one of a kind with plan -review-kind <kind>, and -row <item-id> or -reason <text>", len(pending))
}

// recordOptimizationReview records, in one mutation, the disposition of one
// key or of every pending candidate of one kind: the keys come from state,
// never from a typed list.
func recordOptimizationReview(root, role, key, kind, row, reason string) error {
	return mutatePlan(root, role, "recorded optimization review "+cmp.Or(key, kind), func(document plan.Plan) (plan.Plan, error) {
		keys := []string{strings.TrimSpace(key)}
		if kind != "" {
			store, err := openPlanStore()
			if err != nil {
				return plan.Plan{}, err
			}
			defer store.Close()
			pending, err := pendingOptimizationCandidates(document, store)
			if err != nil {
				return plan.Plan{}, err
			}
			keys = keys[:0]
			for _, candidate := range pending {
				if candidate.Kind == kind {
					keys = append(keys, candidate.Key)
				}
			}
			if len(keys) == 0 {
				return plan.Plan{}, fmt.Errorf("plan: no pending %s candidate to answer", kind)
			}
		}
		for _, each := range keys {
			var err error
			document, err = plan.RecordDisposition(document, plan.OptimizationDisposition{Key: each, Row: strings.TrimSpace(row), Reason: strings.TrimSpace(reason)})
			if err != nil {
				return plan.Plan{}, err
			}
		}
		return document, nil
	})
}
