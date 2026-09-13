package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/runrecord"
)

// candidateProposal names one already-admitted candidate and the plan row it
// should become. The plan door holds no eligibility authority of its own:
// admission happens once, through runrecord.AdmitCandidate, and this spec
// merely cites the decision.
type candidateProposal struct {
	Candidate artifact.ID `json:"candidate"`
	Admission artifact.ID `json:"admission"`
	Slug      string      `json:"slug"`
	Goal      string      `json:"goal"`
	Verify    string      `json:"verify"`
}

// admitProposal appends the plan row one admitted candidate becomes. The
// admission is replayed through the common entry with the core adapters --
// a spec citing a decision the replay cannot reproduce is refused, so the
// plan door can never mint steering eligibility on its own.
func admitProposal(root, specPath string, output io.Writer) error {
	return withPlanMutation(root, false, func(document plan.Plan) error {
		var proposal candidateProposal
		if err := jsonfile.Decode(specPath, &proposal); err != nil {
			return err
		}
		if proposal.Slug == "" || proposal.Goal == "" || proposal.Verify == "" {
			return fmt.Errorf("plan: proposal requires slug, goal, and verify")
		}
		for _, item := range document.Items {
			if item.ID == proposal.Slug {
				return fmt.Errorf("plan: proposal slug %q collides with an open item", proposal.Slug)
			}
		}
		store, err := overgodb.Open(filepath.Join(root, "overgodb-store"))
		if err != nil {
			return err
		}
		defer store.Close()
		ctx := context.Background()
		candidate, err := modelrecipe.RequireCandidate(ctx, store, proposal.Candidate)
		if err != nil {
			return err
		}
		// The replay set is the core adapters plus the composition owner's:
		// alignment-ladder candidates become plan rows through the same
		// door as every other domain.
		adapters := append(modelrecipe.CandidateAdmissionAdapters(), composition.CandidateAdmissionAdapter())
		admission, err := runrecord.RequireReplayedCandidateAdmission(
			ctx, store, proposal.Admission, candidate, adapters...,
		)
		if err != nil {
			return err
		}
		prediction := candidate.Spec().Prediction
		row := plan.Item{
			ID: proposal.Slug, Title: proposal.Goal, Status: plan.StatusOpen,
			Steps: []plan.Step{{
				ID: "do", Title: proposal.Goal, Status: plan.StatusOpen,
				Verify: proposal.Verify,
				Rationale: fmt.Sprintf(
					"candidate %s admitted as %s: predicts %s benefit %g at cost %d %s; uncertainty %g",
					candidate.ID(), admission.ID, prediction.Metric, prediction.Benefit,
					prediction.Cost, prediction.Unit, prediction.Uncertainty,
				),
			}},
		}
		probe := document
		probe.Items = append([]plan.Item{row}, document.Items...)
		if _, err := plan.ResolveCompletionAuthority(ctx, root, "HEAD", probe, store); err != nil {
			return err
		}
		document.Items = append([]plan.Item{row}, document.Items...)
		if err := savePlanMutation(root, document); err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "admitted candidate %s as plan row %s\n", candidate.ID(), row.ID)
		return err
	})
}

// printAttemptHistory renders the store's attempt measurements for
// steering: one strategy-aware summary row per plan step, then the
// matched attempts with their accumulated direction observations, so a
// repeated unsuccessful direction surfaces its pivot recommendation in
// the same readout. The selector "all" admits every item.
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
	fmt.Fprintln(output, "step                                attempts  ok  total wall  churn  recoveries")
	for _, step := range runrecord.SummarizeAttemptHistory(history.Attempts, runrecord.AttemptHistoryFilter{}) {
		fmt.Fprintf(output, "%-34s %8d %3d %11s %6d %11d\n",
			step.PlanItem+"/"+step.PlanStep, step.Attempts, step.Succeeded,
			time.Duration(step.WallNS).Round(time.Millisecond).String(), step.Churn, step.Recoveries)
	}
	fmt.Fprintln(output)
	var directions []runrecord.AttemptDirection
	for _, attempt := range history.Attempts {
		outcome := string(attempt.Outcome)
		if attempt.Failure != "" {
			outcome += ":" + attempt.Failure
		}
		if attempt.Strategy != "" {
			outcome += " strategy=" + attempt.Strategy
		}
		if attempt.BaseManifest.Valid() && attempt.CandidateManifest.Valid() &&
			(attempt.Outcome == runrecord.OutcomeSucceeded || attempt.Outcome == runrecord.OutcomeFailed) {
			direction, directionErr := runrecord.ObserveAttemptDirection(runrecord.AttemptDirectionInput{
				Strategy: attempt.StrategyID, BaseManifest: attempt.BaseManifest,
				CandidateManifest: attempt.CandidateManifest, Objective: attempt.TaskContract,
			}, attempt, directions)
			if directionErr == nil {
				directions = append(directions, direction)
				if direction.Pivot {
					outcome += fmt.Sprintf(" pivot=%q after %d", direction.PivotReason, direction.Count)
				}
			}
		}
		fmt.Fprintf(output, "%s/%s %s wall=%s commit=%.8s selected=%d/%d files=%d\n",
			attempt.PlanItem, attempt.PlanStep, outcome,
			time.Duration(attempt.WallNS).Round(time.Millisecond).String(), attempt.CodeCommit,
			attempt.Selection.Selected, attempt.Selection.Defined, attempt.Diff.Files)
	}
	return nil
}
