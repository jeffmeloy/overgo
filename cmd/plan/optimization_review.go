package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// requireOptimizationReview refuses dispatch while the last landed
// completion's optimization candidates lack a disposition, printing each so
// the re-plan can answer it.
func requireOptimizationReview(document plan.Plan, printLines bool, output io.Writer) error {
	store, err := overgodb.OpenReadOnly(filepath.Join(commandWorktree, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return err
	}
	defer store.Close()
	candidates, err := plan.ReviewLandedCompletion(context.Background(), store, commandWorktree, "HEAD")
	if err != nil {
		return err
	}
	pending := plan.UnreviewedCandidates(candidates, document)
	if len(pending) == 0 {
		return nil
	}
	if printLines {
		for _, line := range plan.FormatCandidates(pending) {
			fmt.Fprintln(output, line)
		}
	}
	return fmt.Errorf("plan: optimization review pending for %d candidate(s) of the landed completion; record each with plan -review <key> -row <item-id> or plan -review <key> -reason <text>", len(pending))
}

// recordOptimizationReview records one disposition through the shared
// mutation owner.
func recordOptimizationReview(root, role, key, row, reason string) error {
	review := plan.OptimizationDisposition{Key: strings.TrimSpace(key), Row: strings.TrimSpace(row), Reason: strings.TrimSpace(reason)}
	return mutatePlan(root, role, "recorded optimization review "+review.Key, func(document plan.Plan) (plan.Plan, error) {
		return plan.RecordDisposition(document, review)
	})
}
