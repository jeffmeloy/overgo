package runrecord

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// OperationalTransitionMediaType is the wire media type.
	OperationalTransitionMediaType = "application/vnd.overgo.operational-transition+json"
	// OperationalTransitionSchema is the versioned schema name.
	OperationalTransitionSchema = "overgo/operational-transition/v1"
)

// TransitionKind names the durable consequences operational state may
// convert into. Raw signals -- heartbeats, wakeups, polls, partial
// output -- never commit; when one matters, it converts here.
type TransitionKind string

// The five conversion kinds the storage boundary admits.
const (
	// TransitionAttempt records an execution attempt becoming durable.
	TransitionAttempt TransitionKind = "attempt"
	// TransitionLease records a work or serving lease transition.
	TransitionLease TransitionKind = "lease"
	// TransitionLoss records admitted work or state being lost.
	TransitionLoss TransitionKind = "loss"
	// TransitionRecovery records lost work being repaired.
	TransitionRecovery TransitionKind = "recovery"
	// TransitionCapabilityChange records a capability catalog change.
	TransitionCapabilityChange TransitionKind = "capability-change"
)

var transitionKinds = []TransitionKind{
	TransitionAttempt, TransitionLease, TransitionLoss,
	TransitionRecovery, TransitionCapabilityChange,
}

// OperationalTransition is the one door operational state passes to
// become canonical: a typed durable consequence binding the subject it
// concerns, the authority that admits it, and bounded evidence.
type OperationalTransition struct {
	Version   uint16         `json:"version"`
	Kind      TransitionKind `json:"kind"`
	Subject   artifact.ID    `json:"subject"`
	Authority artifact.ID    `json:"authority"`
	Evidence  artifact.ID    `json:"evidence,omitzero"`
	Note      string         `json:"note"`
	ID        artifact.ID    `json:"-"`
}

var operationalTransitionCodec = artifact.JSONDocumentCodec(
	"operational transition", artifact.KindEvidence, OperationalTransitionMediaType, OperationalTransitionSchema,
	canonicalizeOperationalTransition,
	func(value OperationalTransition) artifact.ID { return value.ID },
	func(value *OperationalTransition, id artifact.ID) { value.ID = id }, nil,
)

func canonicalizeOperationalTransition(value *OperationalTransition) error {
	if value.Version != artifact.InitialDocumentVersion {
		return errors.New("run record: invalid operational transition version")
	}
	if !slices.Contains(transitionKinds, value.Kind) {
		return errors.New("run record: unknown operational transition kind")
	}
	if !value.Subject.Valid() || value.Authority.Kind() != artifact.KindEvidence {
		return errors.New("run record: operational transition must bind subject and authority")
	}
	if value.Evidence.Valid() && value.Evidence.Kind() != artifact.KindEvidence {
		return errors.New("run record: operational transition evidence must be evidence")
	}
	if value.Note == "" || strings.TrimSpace(value.Note) != value.Note {
		return errors.New("run record: invalid operational transition note")
	}
	return nil
}

// NewOperationalTransition canonicalizes and identifies one conversion.
func NewOperationalTransition(value OperationalTransition) (OperationalTransition, error) {
	value.Version = artifact.InitialDocumentVersion
	return operationalTransitionCodec.New(value)
}

// Content returns the canonical committed bytes of the transition.
func (value OperationalTransition) Content() (artifact.Content, error) {
	return operationalTransitionCodec.Content(value)
}

// Lineage binds the transition to its subject, authority, and evidence.
func (value OperationalTransition) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Subject, value.Authority}
	if value.Evidence.Valid() {
		parents = append(parents, value.Evidence)
	}
	return artifact.DependencyLineage(value.ID, parents...)
}

// PublishOperationalTransition converts one operational event into its
// canonical record as one atomic batch. This is the single door: every
// surface that owes the store a durable transition publishes here, and
// the store itself refuses coordination-shaped schemas that try to
// commit directly.
func PublishOperationalTransition(
	ctx context.Context,
	repository artifact.Repository,
	value OperationalTransition,
) (OperationalTransition, error) {
	admitted, err := NewOperationalTransition(value)
	if err != nil {
		return OperationalTransition{}, err
	}
	content, err := admitted.Content()
	if err != nil {
		return OperationalTransition{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"operational-transition/"+admitted.ID.String(),
		[]artifact.Content{content},
		admitted.Lineage(),
		nil,
	)
	if err != nil {
		return OperationalTransition{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return OperationalTransition{}, err
	}
	return admitted, nil
}
