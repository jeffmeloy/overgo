package plan

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/strictjson"
)

// ProofHorizonPath is the committed receipt that bounds the completion
// authority.
const ProofHorizonPath = "docs/plan_proof_horizon.json"

// ProofHorizon names a commit at or before which completions are accepted
// from Git alone. That commit's hash pins every ancestor, so what a gate
// proved there cannot change afterwards; Completions is how many prepared
// completions it proved, which every resolution recounts.
type ProofHorizon struct {
	Commit      string `json:"commit"`
	Completions int    `json:"completions"`
}

// provenCompletionCommits reads the horizon from the tree of revision --
// never the working tree, so an uncommitted edit skips no proof -- requires
// its commit in that revision's history, and returns the commits at or
// before it. A tree that does not yield the receipt has no horizon and
// proves everything: failing to read it can only ask for more proof.
func provenCompletionCommits(ctx context.Context, repository, revision string, history []gitCompletionMessage) (ProofHorizon, map[string]bool, error) {
	data, err := gitCompletionCommand(ctx, repository, "show", revision+":"+ProofHorizonPath)
	if err != nil {
		return ProofHorizon{}, nil, nil
	}
	var horizon ProofHorizon
	if err := strictjson.DecodeBytes(data, &horizon); err != nil {
		return ProofHorizon{}, nil, fmt.Errorf("plan: proof horizon: %w", err)
	}
	if !slices.ContainsFunc(history, func(commit gitCompletionMessage) bool { return commit.hash == horizon.Commit }) {
		return ProofHorizon{}, nil, fmt.Errorf("plan: proof horizon %.12s is not in the history of %.12s", horizon.Commit, revision)
	}
	ancestors, err := gitCompletionCommand(ctx, repository, "rev-list", horizon.Commit)
	if err != nil {
		return ProofHorizon{}, nil, err
	}
	proven := make(map[string]bool)
	for hash := range strings.FieldsSeq(string(ancestors)) {
		proven[hash] = true
	}
	return horizon, proven, nil
}
