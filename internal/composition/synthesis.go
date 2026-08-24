package composition

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// SynthesisOutcome: terminal decision and evidence.
type SynthesisOutcome struct {
	Decision recipe.Decision
	Record   runrecord.GenerationRecord
	Result   Result
}

// SynthesizeBridge: blocked proposal to measured terminal decision.
func SynthesizeBridge(
	store *overgodb.Store,
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
	target, err := identifyDenseModel(config.TargetDir)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	donor, err := identifyDenseModel(config.DonorDir)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	if target != proposal.Target {
		return SynthesisOutcome{}, errors.New("composition: proposal target differs from synthesis target")
	}
	selected := BridgeCandidate{}
	for _, candidate := range proposal.Candidates {
		if candidate.Donor == donor {
			selected = candidate
			break
		}
	}
	if !selected.Donor.Valid() {
		return SynthesisOutcome{}, errors.New("composition: proposal does not admit the synthesis donor")
	}
	config.DonorTensor = selected.Component
	result, err := runViability(config, target, donor)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	record, err := RecordViability(store, config, result)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	outcome := recipe.DecisionRefused
	tier := recipe.EvidenceExperimental
	reason := result.Reason
	if result.Ship {
		outcome = recipe.DecisionAccepted
		tier = recipe.EvidenceParity
		reason = "bridge synthesis produced held-out gain: " + result.Reason
	}
	decision, err := recipe.NewDecision(
		proposal.ID, outcome, tier, reason, decider,
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
