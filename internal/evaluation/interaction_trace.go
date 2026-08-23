package evaluation

import (
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// InteractionTraceComparison records exact workflow replay dimensions.
type InteractionTraceComparison struct {
	Baseline       artifact.ID `json:"baseline"`
	Candidate      artifact.ID `json:"candidate"`
	Recipe         artifact.ID `json:"recipe"`
	Model          artifact.ID `json:"model"`
	RequestExact   bool        `json:"request_exact"`
	EventsExact    bool        `json:"events_exact"`
	ActionsExact   bool        `json:"actions_exact"`
	DecisionsExact bool        `json:"decisions_exact"`
	ArtifactsExact bool        `json:"artifacts_exact"`
	Exact          bool        `json:"exact"`
	ID             artifact.ID `json:"-"`
}

// CompareInteractionTraces binds one evaluation to two compatible workflow traces.
func CompareInteractionTraces(baseline, candidate runrecord.InteractionTrace) (InteractionTraceComparison, error) {
	if baseline.ValidateIdentity() != nil || candidate.ValidateIdentity() != nil ||
		baseline.Recipe != candidate.Recipe || baseline.Model != candidate.Model {
		return InteractionTraceComparison{}, errors.New("evaluation: interaction traces use different workflow authority")
	}
	eventsExact, err := sameTraceValue(baseline.Events, candidate.Events)
	if err != nil {
		return InteractionTraceComparison{}, err
	}
	comparison := InteractionTraceComparison{
		Baseline: baseline.ID, Candidate: candidate.ID, Recipe: baseline.Recipe, Model: baseline.Model,
		RequestExact: baseline.Request == candidate.Request, EventsExact: eventsExact,
		ActionsExact:   slices.Equal(baseline.ToolActions, candidate.ToolActions),
		DecisionsExact: slices.Equal(baseline.Decisions, candidate.Decisions),
		ArtifactsExact: slices.Equal(baseline.FinalArtifacts, candidate.FinalArtifacts),
	}
	comparison.Exact = comparison.RequestExact && comparison.EventsExact && comparison.ActionsExact &&
		comparison.DecisionsExact && comparison.ArtifactsExact
	id, err := artifact.JSONID(artifact.KindEvaluation, comparison)
	comparison.ID = id
	return comparison, err
}

func sameTraceValue(left, right any) (bool, error) {
	leftID, err := artifact.JSONID(artifact.KindEvidence, left)
	if err != nil {
		return false, err
	}
	rightID, err := artifact.JSONID(artifact.KindEvidence, right)
	return leftID == rightID, err
}
