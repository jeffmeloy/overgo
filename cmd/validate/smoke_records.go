package main

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/inferencesurface"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// A DNA model's inference evidence has no long-form guard: the guard reads a
// text corpus for continuation quality, which says nothing of a genome. Its
// owner is the smoke lane's native-likelihood oracle, which runs the model
// against its exact reference cases and records a bound run at the commit it
// ran on. That run is current while no inference surface source has changed
// since that commit; a change to any of them, the same rule the text guard
// applies through its surface digest, expires it.

// dnaModality is the declared evaluation domain of a DNA model.
const dnaModality = "dna"

// smokeRerun is how a DNA model's smoke run is re-acquired.
const smokeRerun = "run: go run ./cmd/smoke-lane -budget 15m -model %s"

// latestSmokeRun finds the newest bound run of the model's active recipe that
// took the model as an input: the smoke lane's record of serving it.
func latestSmokeRun(ctx context.Context, store *overgodb.Store, model, recipe artifact.ID) (runrecord.Run, bool, error) {
	var latest runrecord.Run
	found := false
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindRun, MediaType: runrecord.RunMediaType, Schema: runrecord.RunSchema}},
		Order:     overgodb.DocumentNewestFirst,
	}, runrecord.ParseRun, func(_ overgodb.DocumentView, run runrecord.Run) error {
		if found || run.Recipe != recipe || !slices.Contains(run.Inputs, model) {
			return nil
		}
		latest, found = run, true
		return nil
	})
	return latest, found, err
}

// changedSince lists the repository-relative paths the working tree differs
// in from a commit, so a run is judged against the tree as it is now.
func changedSince(ctx context.Context, root, commit string) ([]string, error) {
	out, err := gitauthority.Query(ctx, root, "diff", "--name-only", commit)
	if err != nil {
		return nil, err
	}
	var paths []string
	for line := range strings.SplitSeq(strings.TrimSpace(string(out)), "\n") {
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			paths = append(paths, trimmed)
		}
	}
	return paths, nil
}

// smokeCurrent says why a DNA model's smoke run is not current at this tree:
// there is none, it failed, or an inference surface source changed since the
// commit it ran on. Nil means the run stands for the code as it is now.
func smokeCurrent(ctx context.Context, root string, model artifact.ID, run runrecord.Run, found bool) error {
	rerun := fmt.Sprintf(smokeRerun, model)
	switch {
	case !found:
		return fmt.Errorf("no smoke run for this model; %s", rerun)
	case run.Outcome != runrecord.OutcomeSucceeded:
		return fmt.Errorf("its smoke run at commit %.12s failed (%s); fix the runtime, then %s", run.CodeCommit, run.Failure, rerun)
	case run.CodeCommit == "":
		return fmt.Errorf("its smoke run recorded no commit; %s", rerun)
	}
	changed, err := changedSince(ctx, root, run.CodeCommit)
	if err != nil {
		return err
	}
	moved, err := inferencesurface.Moves(ctx, root, changed)
	if err != nil {
		return err
	}
	if len(moved) != 0 {
		return fmt.Errorf("the inference surface moved since its smoke run at commit %.12s (%s); %s", run.CodeCommit, strings.Join(moved, ", "), rerun)
	}
	return nil
}
