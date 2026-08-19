package composition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

// SynthesisOutcome: decision, run record, and measurements.
type SynthesisOutcome struct {
	Decision recipe.Decision
	Record   runrecord.GenerationRecord
	Result   Result
}

// SynthesizeBridge: blocked proposal to measured decision evidence.
func SynthesizeBridge(
	store *repodb.Store,
	proposal BridgeProposal,
	config Config,
	decider recipe.Decider,
) (SynthesisOutcome, error) {
	if store == nil {
		return SynthesisOutcome{}, errors.New("composition: synthesis requires a store")
	}
	if !proposal.ID.Valid() {
		return SynthesisOutcome{}, errors.New("composition: synthesis candidate lacks identity")
	}
	if proposal.State != ProposalPromotionBlocked {
		return SynthesisOutcome{}, errors.New("composition: synthesis input must be a blocked proposal")
	}
	result, err := RunViability(config)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	record, err := RecordViability(store, config, result)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	outcome := recipe.DecisionRefused
	reason := result.Reason
	if result.Ship {
		outcome = recipe.DecisionObserved
		reason = "bridge synthesis produced held-out gain: " + result.Reason
	}
	decision, err := recipe.NewDecision(
		proposal.ID, outcome, recipe.EvidenceExperimental, reason, decider,
		[]artifact.ID{record.ID, record.Run, record.Decision},
	)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	batch, err := decision.Batch("bridge-synthesis/decision/" + decision.ID.String())
	if err != nil {
		return SynthesisOutcome{}, err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		return SynthesisOutcome{}, err
	}
	return SynthesisOutcome{Decision: decision, Record: record, Result: result}, nil
}
