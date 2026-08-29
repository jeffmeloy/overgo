package runrecord

import (
	"context"
	"errors"
	"sort"
	"time"

	"overgo/internal/artifact"
)

// DefaultHeartbeatStaleAfter bounds silence before a running gate is classified stale.
const DefaultHeartbeatStaleAfter = 30 * time.Second

const (
	GateLifecycleMediaType = "application/vnd.overgo.gate-lifecycle+json"
	GateLifecycleSchema    = "overgo/gate-lifecycle/v1"
)

type GateLifecycleState string

const (
	GatePrepared  GateLifecycleState = "prepared"
	GateFinalized GateLifecycleState = "finalized"
)

var gateLifecycleCodec = artifact.JSONDocumentCodec(
	"gate lifecycle", artifact.KindEvidence, GateLifecycleMediaType, GateLifecycleSchema, canonicalizeGateLifecycle,
	func(value GateLifecycle) artifact.ID { return value.ID },
	func(value *GateLifecycle, id artifact.ID) { value.ID = id }, nil,
)

// GateLifecycle is an immutable two-phase record. A preparation is committed
// before Git can advance; its absence from the finalized set is authoritative
// record debt even if the advisory heartbeat disappears.
type GateLifecycle struct {
	Version     uint16             `json:"version"`
	State       GateLifecycleState `json:"state"`
	TreeKey     string             `json:"tree_key"`
	Environment artifact.ID        `json:"environment"`
	Started     string             `json:"started"`
	Preparation *artifact.ID       `json:"preparation,omitempty"`
	CodeCommit  string             `json:"code_commit,omitzero"`
	Result      *artifact.ID       `json:"result,omitempty"`
	Outcome     Outcome            `json:"outcome,omitzero"`
	ID          artifact.ID        `json:"-"`
}

func NewGatePreparation(treeKey string, environment artifact.ID, started time.Time) (GateLifecycle, error) {
	return gateLifecycleCodec.New(GateLifecycle{
		Version: artifact.InitialDocumentVersion, State: GatePrepared, TreeKey: treeKey,
		Environment: environment, Started: started.UTC().Format(time.RFC3339Nano),
	})
}

func NewGateFinalization(preparation GateLifecycle, codeCommit string, result artifact.ID, outcome Outcome) (GateLifecycle, error) {
	if preparation.State != GatePrepared || preparation.ID.Kind() != artifact.KindEvidence {
		return GateLifecycle{}, errors.New("run record: finalization requires an identified preparation")
	}
	preparationID := preparation.ID
	resultID := result
	return gateLifecycleCodec.New(GateLifecycle{
		Version: artifact.InitialDocumentVersion, State: GateFinalized, TreeKey: preparation.TreeKey,
		Environment: preparation.Environment, Started: preparation.Started,
		Preparation: &preparationID, CodeCommit: codeCommit, Result: &resultID, Outcome: outcome,
	})
}

func ParseGateLifecycle(content []byte) (GateLifecycle, error) {
	return gateLifecycleCodec.Parse(content)
}

// RequireGateLifecycle loads one exact typed gate lifecycle document.
func RequireGateLifecycle(ctx context.Context, reader artifact.Reader, id artifact.ID) (GateLifecycle, error) {
	return gateLifecycleCodec.Require(ctx, reader, id)
}

func (l GateLifecycle) Content() (artifact.Content, error) {
	return gateLifecycleCodec.Content(l)
}

func (l GateLifecycle) Lineage() []artifact.Lineage {
	if l.State == GatePrepared {
		return []artifact.Lineage{{Child: l.ID, Parent: l.Environment, Relation: artifact.RelationDependsOn}}
	}
	return []artifact.Lineage{
		{Child: l.ID, Parent: *l.Preparation, Relation: artifact.RelationDerivedFrom},
		{Child: l.ID, Parent: *l.Result, Relation: artifact.RelationDependsOn},
	}
}

func canonicalizeGateLifecycle(lifecycle *GateLifecycle) error {
	if lifecycle == nil || lifecycle.Version != artifact.InitialDocumentVersion ||
		!validTreeKey(lifecycle.TreeKey) ||
		lifecycle.Environment.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid gate lifecycle")
	}
	if _, err := time.Parse(time.RFC3339Nano, lifecycle.Started); err != nil {
		return errors.New("run record: invalid gate lifecycle start")
	}
	switch lifecycle.State {
	case GatePrepared:
		if lifecycle.Preparation != nil || lifecycle.CodeCommit != "" || lifecycle.Result != nil || lifecycle.Outcome != "" {
			return errors.New("run record: prepared gate carries final facts")
		}
	case GateFinalized:
		if lifecycle.Preparation == nil || lifecycle.Preparation.Kind() != artifact.KindEvidence || !validCodeCommit(lifecycle.CodeCommit) ||
			lifecycle.Result == nil || lifecycle.Result.Kind() != artifact.KindEvidence ||
			lifecycle.Outcome != OutcomeSucceeded && lifecycle.Outcome != OutcomeFailed && lifecycle.Outcome != OutcomeCancelled {
			return errors.New("run record: finalized gate lacks final facts")
		}
	default:
		return errors.New("run record: invalid gate lifecycle state")
	}
	return nil
}

// OutstandingGateDebt derives preparations lacking a finalization.
func OutstandingGateDebt(records []GateLifecycle) ([]GateLifecycle, error) {
	prepared := map[artifact.ID]GateLifecycle{}
	var finalizations []GateLifecycle
	for _, lifecycle := range records {
		if err := lifecycle.ValidateIdentity(); err != nil {
			return nil, err
		}
		if lifecycle.State == GatePrepared {
			prepared[lifecycle.ID] = lifecycle
		} else {
			finalizations = append(finalizations, lifecycle)
		}
	}
	finalized := map[artifact.ID]bool{}
	for _, lifecycle := range finalizations {
		preparation, ok := prepared[*lifecycle.Preparation]
		if !ok {
			continue
		}
		if lifecycle.TreeKey != preparation.TreeKey || lifecycle.Environment != preparation.Environment || lifecycle.Started != preparation.Started {
			return nil, errors.New("run record: gate finalization contradicts its preparation")
		}
		finalized[preparation.ID] = true
	}
	debt := make([]GateLifecycle, 0, len(prepared))
	for id, lifecycle := range prepared {
		if !finalized[id] {
			debt = append(debt, lifecycle)
		}
	}
	sort.Slice(debt, func(i, j int) bool { return debt[i].ID.String() < debt[j].ID.String() })
	return debt, nil
}

// ValidateIdentity verifies lifecycle content identity.
func (l GateLifecycle) ValidateIdentity() error { return gateLifecycleCodec.ValidateIdentity(l) }

type GateHeartbeatState string

const (
	HeartbeatRunning    GateHeartbeatState = "running"
	HeartbeatFinalized  GateHeartbeatState = "finalized"
	HeartbeatRecordDebt GateHeartbeatState = "record_debt"
	HeartbeatStale      GateHeartbeatState = "stale"
	HeartbeatAbsent     GateHeartbeatState = "absent"
)

// GateHeartbeat is an advisory liveness mirror. OvergoDB lifecycle documents,
// not this mutable file, determine whether record debt exists.
type GateHeartbeat struct {
	Version     uint16             `json:"version"`
	State       GateHeartbeatState `json:"state"`
	Preparation artifact.ID        `json:"preparation"`
	TreeKey     string             `json:"tree_key"`
	Environment artifact.ID        `json:"environment"`
	PID         int                `json:"pid"`
	Updated     time.Time          `json:"updated"`
}

func (heartbeat GateHeartbeat) Validate() error {
	if heartbeat.Version != artifact.InitialDocumentVersion ||
		heartbeat.State != HeartbeatRunning && heartbeat.State != HeartbeatFinalized && heartbeat.State != HeartbeatRecordDebt ||
		heartbeat.Preparation.Kind() != artifact.KindEvidence || heartbeat.Environment.Kind() != artifact.KindEvidence ||
		!validTreeKey(heartbeat.TreeKey) ||
		heartbeat.PID <= 0 || heartbeat.Updated.IsZero() {
		return errors.New("run record: invalid gate heartbeat")
	}
	return nil
}

func (heartbeat GateHeartbeat) Watchdog(now time.Time, staleAfter time.Duration) GateHeartbeatState {
	if heartbeat.State != HeartbeatRunning {
		return heartbeat.State
	}
	if staleAfter <= 0 || heartbeat.Updated.IsZero() || now.Sub(heartbeat.Updated) > staleAfter {
		return HeartbeatStale
	}
	return HeartbeatRunning
}
