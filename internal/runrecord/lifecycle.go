package runrecord

import (
	"context"
	"errors"
	"fmt"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// DefaultHeartbeatStaleAfter bounds silence before a running gate is classified stale.
const DefaultHeartbeatStaleAfter = 30 * time.Second

const (
	gateLifecycleMediaType = "application/vnd.overgo.gate-lifecycle+json"
	GateLifecycleSchema    = "overgo/gate-lifecycle/v1"
	// gateLifecycleCurrentAlias is the store-local lifecycle head. Its live
	// presence deliberately prevents compaction from rewriting physical gate
	// preparation and finalization receipts.
	gateLifecycleCurrentAlias = overgodb.StoreLocalAliasPrefix + "gate-lifecycle/current"
)

type gateLifecycleState string

const (
	gatePrepared  gateLifecycleState = "prepared"
	gateFinalized gateLifecycleState = "finalized"
)

var gateLifecycleContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: gateLifecycleMediaType, Schema: GateLifecycleSchema,
}

var gateLifecycleCodec = artifact.JSONDocumentCodec(
	"gate lifecycle", gateLifecycleContract.Kind, gateLifecycleContract.MediaType, gateLifecycleContract.Schema, canonicalizeGateLifecycle,
	func(value gateLifecycle) artifact.ID { return value.ID },
	func(value *gateLifecycle, id artifact.ID) { value.ID = id }, nil,
)

// gateLifecycle is an immutable two-phase record. A preparation is committed
// before Git can advance; its absence from the finalized set is authoritative
// record debt even if the advisory heartbeat disappears.
type gateLifecycle struct {
	Version     uint16             `json:"version"`
	State       gateLifecycleState `json:"state"`
	TreeKey     string             `json:"tree_key"`
	Environment artifact.ID        `json:"environment"`
	Started     string             `json:"started"`
	Preparation *artifact.ID       `json:"preparation,omitempty"`
	CodeCommit  string             `json:"code_commit,omitzero"`
	Result      *artifact.ID       `json:"result,omitempty"`
	Outcome     Outcome            `json:"outcome,omitzero"`
	ID          artifact.ID        `json:"-"`
}

func parseGateLifecycle(content []byte) (gateLifecycle, error) {
	return gateLifecycleCodec.Parse(content)
}

// requireGateLifecycle loads one exact typed gate lifecycle document.
func requireGateLifecycle(ctx context.Context, reader artifact.Reader, id artifact.ID) (gateLifecycle, error) {
	return gateLifecycleCodec.Require(ctx, reader, id)
}

func (l gateLifecycle) Content() (artifact.Content, error) {
	return gateLifecycleCodec.Content(l)
}

func (l gateLifecycle) Lineage() []artifact.Lineage {
	if l.State == gatePrepared {
		return []artifact.Lineage{{Child: l.ID, Parent: l.Environment, Relation: artifact.RelationDependsOn}}
	}
	return []artifact.Lineage{
		{Child: l.ID, Parent: *l.Preparation, Relation: artifact.RelationDerivedFrom},
		{Child: l.ID, Parent: *l.Result, Relation: artifact.RelationDependsOn},
	}
}

func canonicalizeGateLifecycle(lifecycle *gateLifecycle) error {
	if lifecycle == nil || lifecycle.Version != artifact.InitialDocumentVersion ||
		!validTreeKey(lifecycle.TreeKey) ||
		lifecycle.Environment.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid gate lifecycle")
	}
	if _, err := time.Parse(time.RFC3339Nano, lifecycle.Started); err != nil {
		return errors.New("run record: invalid gate lifecycle start")
	}
	switch lifecycle.State {
	case gatePrepared:
		if lifecycle.Preparation != nil || lifecycle.CodeCommit != "" || lifecycle.Result != nil || lifecycle.Outcome != "" {
			return errors.New("run record: prepared gate carries final facts")
		}
	case gateFinalized:
		if lifecycle.Preparation == nil || lifecycle.Preparation.Kind() != artifact.KindEvidence || !validCodeCommit(lifecycle.CodeCommit) ||
			lifecycle.Result == nil || lifecycle.Result.Kind() != artifact.KindEvidence ||
			lifecycle.Outcome != OutcomeSucceeded && lifecycle.Outcome != OutcomeFailed && lifecycle.Outcome != OutcomeCancelled && lifecycle.Outcome != outcomeCheckpoint {
			return errors.New("run record: finalized gate lacks final facts")
		}
	default:
		return errors.New("run record: invalid gate lifecycle state")
	}
	return nil
}

// retiredGateLaneAlias is the old gate's deferred-lane obligation head.
const retiredGateLaneAlias = overgodb.StoreLocalAliasPrefix + "gate-lanes/current"

// RetiredGateAliasesAtRest is the store-local admission a physical store
// rewrite (rebuild or release) needs. The only store-local aliases are the
// old gate's lifecycle and lane obligation heads, and nothing writes either
// any more, so both are at rest; their documents are content-addressed and
// cross the rewrite unchanged. Any other store-local alias is refused as
// unknown.
func RetiredGateAliasesAtRest(name string, _ artifact.ID) error {
	if name == gateLifecycleCurrentAlias || name == retiredGateLaneAlias {
		return nil
	}
	return fmt.Errorf("run record: unknown store-local authority %q", name)
}

// ValidateIdentity verifies lifecycle content identity.
func (l gateLifecycle) ValidateIdentity() error { return gateLifecycleCodec.ValidateIdentity(l) }
