package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/scorecard"
)

// improvement closes the loop each time the plan empties: the scorecard run
// is the promotion evidence, the promoted branch moves only on a holding
// scorecard, and the gaps return as rows of the plan at path plan. root is
// the checkout whose branch is promoted and whose plan mutation lock is
// taken; records is the directory the scorecards are published in.
type improvement struct {
	root, plan, records string
	// measure runs the stale validations promotion waits on and publishes
	// the scorecard; fill runs the stale ones it does not compare.
	measure, fill func() error
}

// improvementFor is the loop's improvement world over a store: this
// checkout's plan, the store's published scorecards, the scorecard run and
// the background validations.
func improvementFor(store string) improvement {
	return improvement{
		root: loopWorktree, plan: plan.Path, records: scorecard.Directory(store),
		measure: func() error { return measureScorecard(store) },
		fill: func() error {
			out, err := runTool("go", "run", "./cmd/validate", "-run", "-background", "-repo", store)
			fmt.Print(out)
			return err
		},
	}
}

// Measure runs the scorecard and returns the record it published and that
// record's baseline. A failed validation returns its error beside the
// record; a run that published nothing returns none.
func (w improvement) Measure() (*scorecard.Record, *scorecard.Record, error) {
	before, err := scorecard.Records(w.records)
	if err != nil {
		return nil, nil, err
	}
	measured := w.measure()
	records, err := scorecard.Records(w.records)
	if err != nil || len(records) == len(before) {
		return nil, nil, errors.Join(measured, err)
	}
	baseline, err := scorecard.Baseline(context.Background(), w.root, records)
	if err != nil {
		return nil, nil, errors.Join(measured, err)
	}
	return &records[len(records)-1], baseline, measured
}

// Promote fast-forwards the promoted branch to commit, creating it on the
// first promotion. A branch holding a commit the measured one does not
// contain refuses, and the move holds only if nothing moved the branch since
// it was read. Promotion never pushes: users receive main when it is pushed.
func (w improvement) Promote(commit string) error {
	ctx := context.Background()
	held, err := gitauthority.Query(ctx, w.root, "for-each-ref", "--format=%(objectname)", scorecard.PromotedBranch)
	if err != nil {
		return err
	}
	promoted := strings.TrimSpace(string(held))
	if promoted == commit {
		return nil
	}
	if promoted != "" {
		if _, err := gitauthority.Query(ctx, w.root, "merge-base", "--is-ancestor", promoted, commit); err != nil {
			return fmt.Errorf("%s holds %.12s, which %.12s does not contain; promotion only fast-forwards: %w", scorecard.PromotedBranch, promoted, commit, err)
		}
	}
	var output bytes.Buffer
	update := gitauthority.WriterArguments("update-ref", "-m", "loop: the scorecard holds", scorecard.PromotedBranch, commit, promoted)
	if err := runToolEnv(gitauthority.RepositoryEnvironment(), nil, &output, &output, append([]string{"git", "-C", w.root}, update...)...); err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(output.String()))
	}
	fmt.Printf("loop: promoted %s to %s\n", scorecard.PromotedBranch, commit)
	return nil
}

// Propose appends a row for each gap the plan does not hold, owned by the
// plan's lane and verified by the gap closing on the scorecard the store
// holds, and reports how many it added.
func (w improvement) Propose(published scorecard.Record, gaps []scorecard.Gap) (added int, err error) {
	lock, err := authoritylock.AcquireContext(context.Background(), w.root)
	if err != nil {
		return 0, err
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	document, err := plan.Load(w.plan)
	if err != nil {
		return 0, err
	}
	for _, gap := range gaps {
		if slices.ContainsFunc(document.Items, func(item plan.Item) bool { return item.ID == gap.Key }) {
			continue
		}
		document.Items = append(document.Items, plan.Item{ID: gap.Key, Title: gap.Title, Owner: document.Lane, Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: proposalStep, Title: gap.Title, Status: plan.StatusOpen,
			Verify: "go run ./cmd/validate -scorecard -gap " + gap.Key,
			Rationale: gap.Detail + "\nServes: measure and find the gap; the loop proposed this row from scorecard " + published.Name +
				". Evidence: the verifier reads the scorecard the store holds, judged against that record's baseline.",
		}}})
		added++
	}
	if added == 0 {
		return 0, nil
	}
	return added, plan.Save(w.plan, document)
}

// proposalStep is the one step a proposed row has.
const proposalStep = "do"

// Fill runs the background validations, the suites the baseline scorecard
// does not compare, after the cycle's decision: their evidence flows
// without delaying or withholding a promotion, and a failure is reported
// here and changes nothing the cycle decided.
func (w improvement) Fill() {
	if err := w.fill(); err != nil {
		fmt.Printf("loop: the background validations failed: %v\n", err)
	}
}

// measureScorecard measures again the stale validations promotion waits on,
// then publishes the scorecard of store with what moved since the previous
// one. A failed validation does not withhold the others' numbers: the
// scorecard publishes regardless and the failure is returned beside it with
// the final error the command wrote to its error file, so the withheld
// promotion names each cell or lane validate -run acquired nothing for, and
// why.
func measureScorecard(store string) error {
	errorFile := filepath.Join(os.TempDir(), fmt.Sprintf("overgo-loop-measure-%d.error", os.Getpid()))
	defer os.Remove(errorFile)
	var failed []error
	for _, command := range [][]string{{"go", "run", "./cmd/validate", "-run", "-repo", store}, {"go", "run", "./cmd/validate", "-scorecard", "-publish", "-repo", store}} {
		_ = os.Remove(errorFile)
		if err := runToolEnv(append(loopEnvironment(), clioptions.ErrorFileEnvironment+"="+errorFile), nil, os.Stdout, os.Stdout, command...); err != nil {
			final, _ := os.ReadFile(errorFile)
			failed = append(failed, fmt.Errorf("loop: the plan is empty, but %s failed: %w: %s", strings.Join(command[1:], " "), err, bytes.TrimSpace(final)))
		}
	}
	return errors.Join(failed...)
}
