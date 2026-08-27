package runrecord

import (
	"context"
	"crypto/ed25519"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
)

const (
	// PeerEnrollmentMediaType identifies approved peer enrollment documents.
	PeerEnrollmentMediaType = "application/vnd.overgo.peer-enrollment+json"
	// PeerEnrollmentSchema identifies the approved peer enrollment contract.
	PeerEnrollmentSchema = "overgo/peer-enrollment/v1"
	// PeerStateMediaType identifies peer administrative state documents.
	PeerStateMediaType = "application/vnd.overgo.peer-administrative-state+json"
	// PeerStateSchema identifies the peer administrative state contract.
	PeerStateSchema = "overgo/peer-administrative-state/v1"
	// PeerHeartbeatMediaType identifies capability-bound heartbeat documents.
	PeerHeartbeatMediaType = "application/vnd.overgo.peer-heartbeat+json"
	// PeerHeartbeatSchema identifies the capability-bound heartbeat contract.
	PeerHeartbeatSchema = "overgo/peer-heartbeat/v1"
	// PeerEnrollmentAliasRoot scopes enrollment identities.
	PeerEnrollmentAliasRoot = "peer/enrollment/"
	// PeerNameAliasRoot scopes unique approved peer names.
	PeerNameAliasRoot = "peer/name/"
	// PeerStateAliasRoot scopes current administrative state by peer identity.
	PeerStateAliasRoot = "peer/state/"
	// PeerHeartbeatAliasRoot scopes current heartbeat leases by peer identity.
	PeerHeartbeatAliasRoot = "peer/heartbeat/"
)

// PeerAdministrativeState controls whether a peer may accept new work.
type PeerAdministrativeState string

const (
	// PeerActive admits new serving work and capability publication.
	PeerActive PeerAdministrativeState = "active"
	// PeerDraining refuses new work while allowing in-flight lifecycle reporting.
	PeerDraining PeerAdministrativeState = "draining"
	// PeerRetired permanently refuses serving and heartbeat renewal.
	PeerRetired PeerAdministrativeState = "retired"
)

// PeerEnrollment binds one approved peer name, environment, and public identity.
type PeerEnrollment struct {
	Version        uint16      `json:"version"`
	Name           string      `json:"name"`
	Environment    artifact.ID `json:"environment"`
	PublicKey      []byte      `json:"public_key"`
	ApprovedBy     string      `json:"approved_by"`
	ApprovedUnixNS int64       `json:"approved_unix_ns"`
	ID             artifact.ID `json:"-"`
}

// PeerState records one immutable administrative transition.
type PeerState struct {
	Version       uint16                  `json:"version"`
	Peer          artifact.ID             `json:"peer"`
	State         PeerAdministrativeState `json:"state"`
	ChangedUnixNS int64                   `json:"changed_unix_ns"`
	Previous      artifact.ID             `json:"previous,omitzero"`
	ID            artifact.ID             `json:"-"`
}

// PeerHeartbeat renews one exact peer capability lease.
type PeerHeartbeat struct {
	Version        uint16      `json:"version"`
	Peer           artifact.ID `json:"peer"`
	Capability     artifact.ID `json:"capability"`
	ObservedUnixNS int64       `json:"observed_unix_ns"`
	ExpiresUnixNS  int64       `json:"expires_unix_ns"`
	Previous       artifact.ID `json:"previous,omitzero"`
	ID             artifact.ID `json:"-"`
}

var peerEnrollmentCodec = artifact.JSONDocumentCodec(
	"peer enrollment", artifact.KindEvidence, PeerEnrollmentMediaType, PeerEnrollmentSchema,
	canonicalizePeerEnrollment,
	func(value PeerEnrollment) artifact.ID { return value.ID },
	func(value *PeerEnrollment, id artifact.ID) { value.ID = id },
	func(value PeerEnrollment) PeerEnrollment {
		value.PublicKey = slices.Clone(value.PublicKey)
		return value
	},
)

var peerStateCodec = artifact.JSONDocumentCodec(
	"peer administrative state", artifact.KindEvidence, PeerStateMediaType, PeerStateSchema,
	canonicalizePeerState,
	func(value PeerState) artifact.ID { return value.ID },
	func(value *PeerState, id artifact.ID) { value.ID = id }, nil,
)

var peerHeartbeatCodec = artifact.JSONDocumentCodec(
	"peer heartbeat", artifact.KindEvidence, PeerHeartbeatMediaType, PeerHeartbeatSchema,
	canonicalizePeerHeartbeat,
	func(value PeerHeartbeat) artifact.ID { return value.ID },
	func(value *PeerHeartbeat, id artifact.ID) { value.ID = id }, nil,
)

// ParsePeerEnrollment decodes one exact enrollment document.
func ParsePeerEnrollment(content []byte) (PeerEnrollment, error) {
	return peerEnrollmentCodec.Parse(content)
}

// PublishPeerEnrollment atomically admits an approved identity in active state.
func PublishPeerEnrollment(
	ctx context.Context,
	repository artifact.Repository,
	value PeerEnrollment,
) (PeerEnrollment, PeerState, error) {
	if ctx == nil || repository == nil {
		return PeerEnrollment{}, PeerState{}, errors.New("run record: peer enrollment repository is absent")
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	enrollment, err := peerEnrollmentCodec.New(value)
	if err != nil {
		return PeerEnrollment{}, PeerState{}, err
	}
	state, err := peerStateCodec.New(PeerState{
		Version: artifact.InitialDocumentVersion, Peer: enrollment.ID,
		State: PeerActive, ChangedUnixNS: enrollment.ApprovedUnixNS,
	})
	if err != nil {
		return PeerEnrollment{}, PeerState{}, err
	}
	enrollmentContent, err := peerEnrollmentCodec.Content(enrollment)
	if err != nil {
		return PeerEnrollment{}, PeerState{}, err
	}
	stateContent, err := peerStateCodec.Content(state)
	if err != nil {
		return PeerEnrollment{}, PeerState{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"peer/enrollment/"+enrollment.ID.String(), []artifact.Content{enrollmentContent, stateContent},
		append(
			artifact.DependencyLineage(enrollment.ID, enrollment.Environment),
			artifact.DependencyLineage(state.ID, enrollment.ID)...,
		),
		[]artifact.AliasBinding{
			{Name: PeerEnrollmentAliasRoot + enrollment.ID.String(), Target: enrollment.ID},
			{Name: PeerNameAliasRoot + enrollment.Name, Target: enrollment.ID},
			{Name: PeerStateAliasRoot + enrollment.ID.String(), Target: state.ID},
		},
	)
	if err != nil {
		return PeerEnrollment{}, PeerState{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return PeerEnrollment{}, PeerState{}, err
	}
	return enrollment, state, nil
}

// RequirePeerEnrollment resolves one peer's immutable approved identity.
func RequirePeerEnrollment(ctx context.Context, reader artifact.Reader, id artifact.ID) (PeerEnrollment, error) {
	return peerEnrollmentCodec.Require(ctx, reader, id)
}

// ResolvePeerState returns the current administrative state for one peer.
func ResolvePeerState(ctx context.Context, reader artifact.Reader, peer artifact.ID) (PeerState, bool, error) {
	return peerStateCodec.Resolve(ctx, reader, PeerStateAliasRoot+peer.String())
}

// PublishPeerState advances active to draining and draining to retired.
func PublishPeerState(
	ctx context.Context,
	repository artifact.Repository,
	peer artifact.ID,
	state PeerAdministrativeState,
	changedUnixNS int64,
) (PeerState, error) {
	if ctx == nil || repository == nil {
		return PeerState{}, errors.New("run record: peer state repository is absent")
	}
	if _, err := RequirePeerEnrollment(ctx, repository, peer); err != nil {
		return PeerState{}, err
	}
	previous, found, err := ResolvePeerState(ctx, repository, peer)
	if err != nil || !found {
		return PeerState{}, errors.Join(errors.New("run record: peer state authority is absent"), err)
	}
	if !peerStateTransition(previous.State, state) || changedUnixNS <= previous.ChangedUnixNS {
		return PeerState{}, errors.New("run record: invalid peer administrative transition")
	}
	value, err := peerStateCodec.New(PeerState{
		Version: artifact.InitialDocumentVersion, Peer: peer, State: state,
		ChangedUnixNS: changedUnixNS, Previous: previous.ID,
	})
	if err != nil {
		return PeerState{}, err
	}
	content, err := peerStateCodec.Content(value)
	if err != nil {
		return PeerState{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"peer/state/"+value.ID.String(), []artifact.Content{content},
		artifact.DependencyLineage(value.ID, peer, previous.ID),
		[]artifact.AliasBinding{{
			Name: PeerStateAliasRoot + peer.String(), Target: value.ID, Previous: artifact.IDPointer(previous.ID),
		}},
	)
	if err != nil {
		return PeerState{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return PeerState{}, err
	}
	return value, nil
}

// ResolvePeerHeartbeat returns the current lease renewal for one peer.
func ResolvePeerHeartbeat(ctx context.Context, reader artifact.Reader, peer artifact.ID) (PeerHeartbeat, bool, error) {
	return peerHeartbeatCodec.Resolve(ctx, reader, PeerHeartbeatAliasRoot+peer.String())
}

// PublishPeerHeartbeat atomically renews one capability-bound lease.
func PublishPeerHeartbeat(
	ctx context.Context,
	repository artifact.Repository,
	value PeerHeartbeat,
) (PeerHeartbeat, error) {
	if ctx == nil || repository == nil || value.Previous.Valid() {
		return PeerHeartbeat{}, errors.New("run record: invalid peer heartbeat publication")
	}
	if _, err := RequirePeerEnrollment(ctx, repository, value.Peer); err != nil {
		return PeerHeartbeat{}, err
	}
	state, found, err := ResolvePeerState(ctx, repository, value.Peer)
	if err != nil || !found || state.State == PeerRetired {
		return PeerHeartbeat{}, errors.Join(errors.New("run record: retired or unknown peer cannot heartbeat"), err)
	}
	previous, hadPrevious, err := ResolvePeerHeartbeat(ctx, repository, value.Peer)
	if err != nil {
		return PeerHeartbeat{}, err
	}
	if hadPrevious {
		if value.ObservedUnixNS <= previous.ObservedUnixNS {
			return PeerHeartbeat{}, errors.New("run record: peer heartbeat does not advance time")
		}
		value.Previous = previous.ID
	}
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	value, err = peerHeartbeatCodec.New(value)
	if err != nil {
		return PeerHeartbeat{}, err
	}
	content, err := peerHeartbeatCodec.Content(value)
	if err != nil {
		return PeerHeartbeat{}, err
	}
	alias := artifact.AliasBinding{Name: PeerHeartbeatAliasRoot + value.Peer.String(), Target: value.ID}
	if hadPrevious {
		alias.Previous = artifact.IDPointer(previous.ID)
	}
	parents := []artifact.ID{value.Peer, value.Capability, value.Previous}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	batch, err := artifact.NewDocumentBatch(
		"peer/heartbeat/"+value.ID.String(), []artifact.Content{content},
		artifact.DependencyLineage(value.ID, parents...), []artifact.AliasBinding{alias},
	)
	if err != nil {
		return PeerHeartbeat{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return PeerHeartbeat{}, err
	}
	return value, nil
}

func canonicalizePeerEnrollment(value *PeerEnrollment) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!peerLabel(value.Name) || value.Environment.Kind() != artifact.KindEvidence ||
		len(value.PublicKey) != ed25519.PublicKeySize || !peerLabel(value.ApprovedBy) || value.ApprovedUnixNS <= 0 {
		return errors.New("run record: invalid peer enrollment")
	}
	value.PublicKey = slices.Clone(value.PublicKey)
	return nil
}

func canonicalizePeerState(value *PeerState) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Peer.Kind() != artifact.KindEvidence ||
		(value.State != PeerActive && value.State != PeerDraining && value.State != PeerRetired) || value.ChangedUnixNS <= 0 ||
		(value.State == PeerActive) != !value.Previous.Valid() || value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid peer administrative state")
	}
	return nil
}

func canonicalizePeerHeartbeat(value *PeerHeartbeat) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Peer.Kind() != artifact.KindEvidence ||
		value.Capability.Kind() != artifact.KindEvidence || value.ObservedUnixNS <= 0 || value.ExpiresUnixNS <= value.ObservedUnixNS ||
		value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid peer heartbeat")
	}
	return nil
}

func peerStateTransition(previous, next PeerAdministrativeState) bool {
	return previous == PeerActive && next == PeerDraining || previous == PeerDraining && next == PeerRetired
}

func peerLabel(value string) bool {
	return value != "" && strings.TrimSpace(value) == value && !strings.ContainsAny(value, " /\\\t\x00\r\n")
}
