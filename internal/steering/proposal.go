// Package steering owns typed steering proposals: the bounded
// interface through which a model -- or a human -- proposes the next
// task from accumulated evidence. A proposal states its goal, its
// predicted benefit and cost, its explicit uncertainty, a falsifiable
// check, and references into the measurement history it reasons from.
// Deterministic admission validates every reference against the store
// and converts an admitted proposal into a plan row carrying the
// proposal as provenance. Proposals never execute anything: admission
// produces a row for the same gate-verified loop every other row runs
// through, and refusal produces nothing.
package steering

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

const (
	proposalMediaType = "application/vnd.overgo.steering-proposal+json"
	proposalSchema    = "overgo/steering-proposal/v1"
	// proposalTextBound caps every free-text field: a proposal is a
	// bounded argument for one task, never an essay or a payload.
	proposalTextBound = 2048
)

// Proposal is one typed steering request.
type Proposal struct {
	ID artifact.ID `json:"-"`
	// Version is the wire version stamp.
	Version uint16 `json:"version"`
	// Slug is the plan item identity the proposal becomes on admission.
	Slug string `json:"slug"`
	// Goal is the task, stated as the row title.
	Goal string `json:"goal"`
	// Proposer identifies the steering source: a strategy identity or
	// a human handle.
	Proposer string `json:"proposer"`
	// PredictedBenefit states what measurably improves if the task
	// succeeds.
	PredictedBenefit string `json:"predicted_benefit"`
	// PredictedCost states the expected spend in the proposer's terms.
	PredictedCost string `json:"predicted_cost"`
	// Uncertainty states explicitly what the proposer does not know.
	Uncertainty string `json:"uncertainty"`
	// FalsifiableCheck is the machine verify command the row will carry:
	// the experiment that can prove the proposal wrong.
	FalsifiableCheck string `json:"falsifiable_check"`
	// History references the measurement records the proposal reasons
	// from; every reference must exist in the store.
	History []artifact.ID `json:"history"`
}

var proposalCodec = artifact.JSONDocumentCodec(
	"steering proposal", artifact.KindEvidence, proposalMediaType, proposalSchema,
	func(value *Proposal) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion {
			return errors.New("steering: invalid proposal version")
		}
		if strings.TrimSpace(value.Slug) == "" || strings.ContainsAny(value.Slug, " /\x00\r\n\t") {
			return errors.New("steering: proposal slug must be one bounded token")
		}
		for name, text := range map[string]string{
			"goal": value.Goal, "proposer": value.Proposer,
			"predicted_benefit": value.PredictedBenefit, "predicted_cost": value.PredictedCost,
			"uncertainty": value.Uncertainty, "falsifiable_check": value.FalsifiableCheck,
		} {
			if strings.TrimSpace(text) == "" || len(text) > proposalTextBound ||
				strings.ContainsAny(text, "\x00") {
				return fmt.Errorf("steering: proposal %s must be present and bounded", name)
			}
		}
		if len(value.History) == 0 {
			return errors.New("steering: a proposal reasons from recorded history or it is a guess")
		}
		for _, id := range value.History {
			if !id.Valid() {
				return errors.New("steering: invalid history reference")
			}
		}
		return nil
	},
	func(value Proposal) artifact.ID { return value.ID },
	func(value *Proposal, id artifact.ID) { value.ID = id },
	func(value Proposal) Proposal {
		value.History = append([]artifact.ID(nil), value.History...)
		return value
	},
)

// Admit validates a proposal deterministically, records it in the
// store, and returns the plan row it becomes. Every history reference
// must already exist in the store: a proposal citing measurements
// nobody recorded is refused, not repaired. The caller owns appending
// the row to the plan; admission itself executes nothing.
func Admit(ctx context.Context, store *overgodb.Store, proposal Proposal) (Proposal, plan.Item, error) {
	proposal.Version = artifact.InitialDocumentVersion
	admitted, err := proposalCodec.New(proposal)
	if err != nil {
		return Proposal{}, plan.Item{}, err
	}
	for _, id := range admitted.History {
		_, _, found, err := store.OpenContent(ctx, id)
		if err != nil {
			return Proposal{}, plan.Item{}, err
		}
		if !found {
			return Proposal{}, plan.Item{}, fmt.Errorf("steering: history reference %s is not in the store", id)
		}
	}
	content, err := proposalCodec.Content(admitted)
	if err != nil {
		return Proposal{}, plan.Item{}, err
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "steering/proposal/" + admitted.ID.String(),
		Artifacts: []artifact.Descriptor{content.Descriptor},
		Contents:  []artifact.Content{content},
	}); err != nil {
		return Proposal{}, plan.Item{}, err
	}
	return admitted, planRow(admitted), nil
}

// planRow renders the admitted proposal as the row the loop will run:
// the falsifiable check is the row's machine verify, and the rationale
// carries the proposal identity with its prediction and uncertainty,
// so the row's eventual outcome is comparable against what was
// predicted.
func planRow(admitted Proposal) plan.Item {
	rationale := fmt.Sprintf(
		"proposal %s by %s: predicts %s at %s; uncertainty: %s",
		admitted.ID, admitted.Proposer, admitted.PredictedBenefit,
		admitted.PredictedCost, admitted.Uncertainty,
	)
	return plan.Item{
		ID: admitted.Slug, Title: admitted.Goal, Status: plan.StatusOpen,
		Steps: []plan.Step{{
			ID: "do", Title: admitted.Goal, Status: plan.StatusOpen,
			Verify: admitted.FalsifiableCheck, Rationale: rationale,
		}},
	}
}
