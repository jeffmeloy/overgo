package composition

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

// SynthesisOutcome: the pipeline's typed result -- the decision (promotion
// evidence or refusal with measured reason), the generation record binding
// the experiment, and the probe measurements behind both.
type SynthesisOutcome struct {
	Decision recipe.Decision
	Record   runrecord.GenerationRecord
	Result   Result
}

// SynthesizeBridge executes the full pipeline for one blocked candidate:
// derive the adapter geometry from the donor component and target (NewGraft's
// derivation inside RunViability), train the bridge with everything else
// frozen under the recorded protocol, commit the generation record, and emit
// a typed decision -- DecisionAccepted carrying the record as promotion
// evidence when the probe ships, DecisionRefused with the measured reason
// otherwise. The proposal's promotion-blocked state is never mutated: the
// decision is new evidence the experiment plane consumes, not an override.
func SynthesizeBridge(
	store *repodb.Store,
	proposal BridgeProposal,
	config Config,
	decider recipe.Decider,
) (SynthesisOutcome, error) {
	if store == nil {
		return SynthesisOutcome{}, fmt.Errorf("composition: synthesis requires a store")
	}
	if !proposal.ID.Valid() {
		return SynthesisOutcome{}, fmt.Errorf("composition: synthesis candidate lacks identity")
	}
	if proposal.State != ProposalPromotionBlocked {
		return SynthesisOutcome{}, fmt.Errorf("composition: synthesis input must be a blocked proposal")
	}
	result, err := RunViability(config)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	record, err := RecordViability(store, config, result)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	outcome, tier := recipe.DecisionRefused, recipe.EvidenceExperimental
	reason := result.Reason
	if result.Ship {
		outcome, tier = recipe.DecisionAccepted, recipe.EvidenceParity
		reason = "bridge synthesis produced held-out gain: " + result.Reason
	}
	decision, err := recipe.NewDecision(
		proposal.ID, outcome, tier, reason, decider,
		[]artifact.ID{record.ID, record.Run, record.Decision},
	)
	if err != nil {
		return SynthesisOutcome{}, err
	}
	content, err := decision.Content()
	if err != nil {
		return SynthesisOutcome{}, err
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:      "bridge-synthesis/decision/" + decision.ID.String(),
		Contents: []artifact.Content{content},
		Lineage: []artifact.Lineage{{
			Child: decision.ID, Parent: proposal.ID, Relation: artifact.RelationDependsOn,
		}},
	}); err != nil {
		return SynthesisOutcome{}, err
	}
	return SynthesisOutcome{Decision: decision, Record: record, Result: result}, nil
}
