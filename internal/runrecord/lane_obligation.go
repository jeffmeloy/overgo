package runrecord

import (
	"context"
	"errors"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// LaneObligationState is where a deferred lane run stands.
type LaneObligationState string

const (
	// LaneObligationPending marks a landed commit whose runner has not started.
	LaneObligationPending LaneObligationState = "pending"
	// LaneObligationRunning marks an obligation a runner holds.
	LaneObligationRunning LaneObligationState = "running"
	// LaneObligationPassed marks every deferred lane passed on the landed commit.
	LaneObligationPassed LaneObligationState = "passed"
	// LaneObligationFailed marks a deferred lane failure; the next gate runs its lanes inline.
	LaneObligationFailed LaneObligationState = "failed"
	// LaneObligationSuperseded marks a debt a later inline lane pass on a descendant resolved.
	LaneObligationSuperseded LaneObligationState = "superseded"
)

const (
	// GateLaneObligationMediaType identifies deferred lane obligations.
	GateLaneObligationMediaType = "application/vnd.overgo.gate-lane-obligation+json"
	// GateLaneObligationSchema identifies the obligation schema.
	GateLaneObligationSchema = "overgo/gate-lane-obligation/v1"
	// GateLaneObligationAlias names the store's current lane obligation.
	GateLaneObligationAlias = overgodb.StoreLocalAliasPrefix + "gate-lanes/current"
)

var gateLaneObligationContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: GateLaneObligationMediaType, Schema: GateLaneObligationSchema,
}

var gateLaneObligationCodec = artifact.JSONDocumentCodec(
	"gate lane obligation", gateLaneObligationContract.Kind, gateLaneObligationContract.MediaType, gateLaneObligationContract.Schema,
	canonicalizeGateLaneObligation,
	func(value GateLaneObligation) artifact.ID { return value.ID },
	func(value *GateLaneObligation, id artifact.ID) { value.ID = id },
	func(value GateLaneObligation) GateLaneObligation {
		value.Checks = slices.Clone(value.Checks)
		value.Paths = slices.Clone(value.Paths)
		return value
	},
)

// GateLaneObligation records the lanes a gate deferred past its commit and
// what became of them. Every state is its own immutable document; the alias
// moves along the chain, and Previous names the state it replaced.
type GateLaneObligation struct {
	Version     uint16              `json:"version"`
	State       LaneObligationState `json:"state"`
	CodeCommit  string              `json:"code_commit"`
	Preparation artifact.ID         `json:"preparation"`
	Result      artifact.ID         `json:"result"`
	Checks      []string            `json:"checks"`
	Paths       []string            `json:"paths"`
	Updated     string              `json:"updated"`
	Outcome     artifact.ID         `json:"outcome,omitzero"`
	Previous    artifact.ID         `json:"previous,omitzero"`
	ID          artifact.ID         `json:"-"`
}

// NewGateLaneObligation records the lanes a landed commit still owes.
func NewGateLaneObligation(codeCommit string, preparation, result artifact.ID, checks, paths []string, now time.Time) (GateLaneObligation, error) {
	return gateLaneObligationCodec.NewInitial(GateLaneObligation{
		State: LaneObligationPending, CodeCommit: codeCommit, Preparation: preparation, Result: result,
		Checks: slices.Clone(checks), Paths: slices.Clone(paths), Updated: now.UTC().Format(time.RFC3339Nano),
	})
}

// Transition derives the next state of the chain from this document.
func (obligation GateLaneObligation) Transition(state LaneObligationState, outcome artifact.ID, now time.Time) (GateLaneObligation, error) {
	if err := obligation.ValidateIdentity(); err != nil {
		return GateLaneObligation{}, err
	}
	if !laneObligationTransition(obligation.State, state) {
		return GateLaneObligation{}, errors.New("run record: lane obligation cannot move from " + string(obligation.State) + " to " + string(state))
	}
	next := obligation
	next.State, next.Outcome, next.Previous = state, outcome, obligation.ID
	next.Updated = now.UTC().Format(time.RFC3339Nano)
	next.ID = artifact.ID{}
	return gateLaneObligationCodec.New(next)
}

// Resolved reports a state that owes nothing further.
func (obligation GateLaneObligation) Resolved() bool {
	return obligation.State == LaneObligationPassed || obligation.State == LaneObligationSuperseded
}

// ValidateIdentity checks canonical form and content-addressed identity.
func (obligation GateLaneObligation) ValidateIdentity() error {
	return gateLaneObligationCodec.ValidateIdentity(obligation)
}

// Content is the document's canonical bytes and descriptor.
func (obligation GateLaneObligation) Content() (artifact.Content, error) {
	return gateLaneObligationCodec.Content(obligation)
}

// Batch publishes the document and moves the current alias onto it by CAS.
func (obligation GateLaneObligation) Batch(previous *artifact.ID) (artifact.Batch, error) {
	var lineage []artifact.Lineage
	if obligation.Previous.Valid() {
		lineage = []artifact.Lineage{{Child: obligation.ID, Parent: obligation.Previous, Relation: artifact.RelationDerivedFrom}}
	}
	return gateLaneObligationCodec.Batch("gate/lanes/"+obligation.ID.String(), obligation, lineage,
		[]artifact.AliasBinding{{Name: GateLaneObligationAlias, Target: obligation.ID, Previous: previous}})
}

// CurrentGateLaneObligation resolves the store's current obligation, if any.
func CurrentGateLaneObligation(ctx context.Context, reader artifact.Reader) (GateLaneObligation, bool, error) {
	return gateLaneObligationCodec.Resolve(ctx, reader, GateLaneObligationAlias)
}

func laneObligationTransition(from, to LaneObligationState) bool {
	switch from {
	case LaneObligationPending:
		return to == LaneObligationRunning || to == LaneObligationSuperseded
	case LaneObligationRunning:
		return to == LaneObligationPassed || to == LaneObligationFailed || to == LaneObligationSuperseded
	case LaneObligationFailed:
		return to == LaneObligationSuperseded || to == LaneObligationRunning
	default:
		return false
	}
}

func canonicalizeGateLaneObligation(value *GateLaneObligation) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !validCodeCommit(value.CodeCommit) ||
		value.Preparation.Kind() != artifact.KindEvidence || value.Result.Kind() != artifact.KindEvidence ||
		len(value.Checks) == 0 || len(value.Paths) == 0 {
		return errors.New("run record: invalid gate lane obligation")
	}
	switch value.State {
	case LaneObligationPending, LaneObligationRunning:
		if value.Outcome.Valid() {
			return errors.New("run record: an open lane obligation has no outcome")
		}
	case LaneObligationPassed, LaneObligationFailed, LaneObligationSuperseded:
		if value.Outcome.Kind() != artifact.KindEvidence {
			return errors.New("run record: a resolved lane obligation names its outcome evidence")
		}
	default:
		return errors.New("run record: invalid lane obligation state")
	}
	if value.State != LaneObligationPending && !value.Previous.Valid() {
		return errors.New("run record: a lane obligation state after pending names the state it replaced")
	}
	if value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return errors.New("run record: lane obligation previous state is not evidence")
	}
	parsed, err := time.Parse(time.RFC3339Nano, value.Updated)
	if err != nil {
		return errors.New("run record: invalid lane obligation timestamp")
	}
	value.Updated = parsed.UTC().Format(time.RFC3339Nano)
	slices.Sort(value.Checks)
	value.Checks = slices.Compact(value.Checks)
	slices.Sort(value.Paths)
	value.Paths = slices.Compact(value.Paths)
	for _, name := range value.Checks {
		if !validLabel(name) {
			return errors.New("run record: invalid lane obligation check name")
		}
	}
	for _, path := range value.Paths {
		if !validText(path) {
			return errors.New("run record: invalid lane obligation path")
		}
	}
	return nil
}
