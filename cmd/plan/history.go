package main

import (
	"context"
	"fmt"
	"io"
	"time"

	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
	"overgo/internal/steering"
)

// admitProposal runs one typed steering proposal through deterministic
// admission: the store records the proposal, and the plan gains the
// row it becomes -- on top, dispatchable, carrying the proposal as its
// rationale. Refusal changes nothing.
func admitProposal(specPath string, document plan.Plan, output io.Writer) error {
	var proposal steering.Proposal
	if err := jsonfile.Decode(specPath, &proposal); err != nil {
		return err
	}
	store, err := overgodb.Open("overgodb-store")
	if err != nil {
		return err
	}
	defer store.Close()
	admitted, row, err := steering.Admit(context.Background(), store, proposal)
	if err != nil {
		return err
	}
	for _, item := range document.Items {
		if item.ID == row.ID {
			return fmt.Errorf("plan: proposal slug %q collides with an open item", row.ID)
		}
	}
	document.Items = append([]plan.Item{row}, document.Items...)
	if err := plan.Save("", document); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "admitted proposal %s as plan row %s\n", admitted.ID, row.ID)
	return err
}

// printAttemptHistory renders the store's attempt measurements for
// steering: one aggregate row per plan step, then the matched
// attempts. The selector "all" admits every item.
func printAttemptHistory(selector string, output io.Writer) error {
	store, err := overgodb.OpenReadOnly("overgodb-store")
	if err != nil {
		return err
	}
	defer store.Close()
	filter := runrecord.AttemptFilter{}
	if selector != "all" {
		filter.PlanItem = selector
	}
	history, err := runrecord.LoadAttemptHistory(context.Background(), store, filter)
	if err != nil {
		return err
	}
	if len(history.Attempts) == 0 {
		_, err = fmt.Fprintln(output, "no attempt records match")
		return err
	}
	fmt.Fprintln(output, "step                                attempts  ok  total wall  files")
	for _, step := range history.Steps {
		fmt.Fprintf(output, "%-34s %8d %3d %11s %6d\n",
			step.PlanItem+"/"+step.PlanStep, step.Attempts, step.Succeeded,
			time.Duration(step.TotalWallNS).Round(time.Millisecond).String(), step.TotalFiles)
	}
	fmt.Fprintln(output)
	for _, attempt := range history.Attempts {
		outcome := string(attempt.Outcome)
		if attempt.Failure != "" {
			outcome += ":" + attempt.Failure
		}
		if attempt.Strategy != "" {
			outcome += " strategy=" + attempt.Strategy
		}
		fmt.Fprintf(output, "%s/%s %s wall=%s commit=%.8s selected=%d/%d files=%d\n",
			attempt.PlanItem, attempt.PlanStep, outcome,
			time.Duration(attempt.WallNS).Round(time.Millisecond).String(), attempt.CodeCommit,
			attempt.Selection.Selected, attempt.Selection.Defined, attempt.Diff.Files)
	}
	return nil
}
