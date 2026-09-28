package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"slices"

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
	return withPlanMutation(root, func(document plan.Plan) error {
		var proposal candidateProposal
		if err := jsonfile.Decode(specPath, &proposal); err != nil {
			return err
		}
		if proposal.Slug == "" || proposal.Goal == "" || proposal.Verify == "" {
			return fmt.Errorf("plan: proposal requires slug, goal, and verify")
		}
		if slices.ContainsFunc(document.Items, itemNamed(proposal.Slug)) {
			return fmt.Errorf("plan: proposal slug %q collides with an open item", proposal.Slug)
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
		document.Items = append([]plan.Item{row}, document.Items...)
		if err := savePlanMutation(root, document); err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "admitted candidate %s as plan row %s\n", candidate.ID(), row.ID)
		return err
	})
}
