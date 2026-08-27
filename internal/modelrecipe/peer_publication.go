package modelrecipe

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	// PeerCapabilityPublicationMediaType identifies peer capability publication documents.
	PeerCapabilityPublicationMediaType = "application/vnd.overgo.peer-capability-publication+json"
	// PeerCapabilityPublicationSchema identifies the peer capability publication contract.
	PeerCapabilityPublicationSchema = "overgo/peer-capability-publication/v1"
	// PeerCapabilityPublicationAliasRoot scopes current declarations by peer identity.
	PeerCapabilityPublicationAliasRoot = "peer/capability/"
)

// PeerCapabilityPublication binds a declared transport to one enrolled peer.
type PeerCapabilityPublication struct {
	Version         uint16      `json:"version"`
	Peer            artifact.ID `json:"peer"`
	Capability      artifact.ID `json:"capability"`
	PublishedUnixNS int64       `json:"published_unix_ns"`
	Previous        artifact.ID `json:"previous,omitzero"`
	ID              artifact.ID `json:"-"`
}

// PeerServingAuthority is the exact enrolled, active, capability-bound lease admitted for serving.
type PeerServingAuthority struct {
	Enrollment  runrecord.PeerEnrollment  `json:"enrollment"`
	State       runrecord.PeerState       `json:"state"`
	Publication PeerCapabilityPublication `json:"publication"`
	Capability  RemotePeerCapability      `json:"capability"`
	Heartbeat   runrecord.PeerHeartbeat   `json:"heartbeat"`
}

var peerCapabilityPublicationCodec = artifact.JSONDocumentCodec(
	"peer capability publication", artifact.KindEvidence,
	PeerCapabilityPublicationMediaType, PeerCapabilityPublicationSchema,
	canonicalizePeerCapabilityPublication,
	func(value PeerCapabilityPublication) artifact.ID { return value.ID },
	func(value *PeerCapabilityPublication, id artifact.ID) { value.ID = id }, nil,
)

// ResolvePeerCapabilityPublication returns the current declaration for one peer.
func ResolvePeerCapabilityPublication(
	ctx context.Context,
	reader artifact.Reader,
	peer artifact.ID,
) (PeerCapabilityPublication, RemotePeerCapability, bool, error) {
	publication, found, err := peerCapabilityPublicationCodec.Resolve(ctx, reader, PeerCapabilityPublicationAliasRoot+peer.String())
	if err != nil || !found {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, found, err
	}
	capability, err := RequireRemotePeerCapability(ctx, reader, publication.Capability)
	return publication, capability, err == nil, err
}

// ResolvePeerServingAuthority requires an active peer and unexpired current capability lease.
func ResolvePeerServingAuthority(
	ctx context.Context,
	repository artifact.Repository,
	peer artifact.ID,
	nowUnixNS int64,
) (PeerServingAuthority, error) {
	if ctx == nil || repository == nil || nowUnixNS <= 0 {
		return PeerServingAuthority{}, errors.New("model recipe: invalid peer serving authority request")
	}
	enrollment, err := runrecord.RequirePeerEnrollment(ctx, repository, peer)
	if err != nil {
		return PeerServingAuthority{}, err
	}
	state, found, err := runrecord.ResolvePeerState(ctx, repository, peer)
	if err != nil || !found || state.State != runrecord.PeerActive {
		return PeerServingAuthority{}, errors.Join(errors.New("model recipe: peer is not active"), err)
	}
	publication, capability, found, err := ResolvePeerCapabilityPublication(ctx, repository, peer)
	if err != nil || !found || capability.Environment != enrollment.Environment {
		return PeerServingAuthority{}, errors.Join(errors.New("model recipe: peer capability is unavailable"), err)
	}
	heartbeat, found, err := runrecord.ResolvePeerHeartbeat(ctx, repository, peer)
	if err != nil || !found || heartbeat.Peer != peer || heartbeat.Capability != publication.ID ||
		heartbeat.ObservedUnixNS > nowUnixNS || nowUnixNS >= heartbeat.ExpiresUnixNS {
		return PeerServingAuthority{}, errors.Join(errors.New("model recipe: peer lease is absent, mismatched, or expired"), err)
	}
	return PeerServingAuthority{
		Enrollment: enrollment, State: state, Publication: publication,
		Capability: capability, Heartbeat: heartbeat,
	}, nil
}

// PublishPeerCapability atomically advances one active peer's declaration.
func PublishPeerCapability(
	ctx context.Context,
	repository artifact.Repository,
	peer artifact.ID,
	capability RemotePeerCapability,
	publishedUnixNS int64,
) (PeerCapabilityPublication, RemotePeerCapability, error) {
	if ctx == nil || repository == nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, errors.New("model recipe: peer capability repository is absent")
	}
	enrollment, err := runrecord.RequirePeerEnrollment(ctx, repository, peer)
	if err != nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, err
	}
	state, found, err := runrecord.ResolvePeerState(ctx, repository, peer)
	if err != nil || !found || state.State != runrecord.PeerActive {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, errors.Join(errors.New("model recipe: peer is not active"), err)
	}
	capability, err = NewRemotePeerCapability(capability)
	if err != nil || capability.Environment != enrollment.Environment {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, errors.Join(errors.New("model recipe: peer capability identity differs"), err)
	}
	previous, _, hadPrevious, err := ResolvePeerCapabilityPublication(ctx, repository, peer)
	if err != nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, err
	}
	publication := PeerCapabilityPublication{
		Version: artifact.InitialDocumentVersion, Peer: peer, Capability: capability.ID,
		PublishedUnixNS: publishedUnixNS,
	}
	if hadPrevious {
		if publishedUnixNS <= previous.PublishedUnixNS {
			return PeerCapabilityPublication{}, RemotePeerCapability{}, errors.New("model recipe: peer capability publication does not advance time")
		}
		publication.Previous = previous.ID
	}
	publication, err = peerCapabilityPublicationCodec.New(publication)
	if err != nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, err
	}
	capabilityContent, err := remotePeerCapabilityCodec.Content(capability)
	if err != nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, err
	}
	publicationContent, err := peerCapabilityPublicationCodec.Content(publication)
	if err != nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, err
	}
	alias := artifact.AliasBinding{Name: PeerCapabilityPublicationAliasRoot + peer.String(), Target: publication.ID}
	if hadPrevious {
		alias.Previous = artifact.IDPointer(previous.ID)
	}
	parents := []artifact.ID{peer, capability.Environment, capability.ID}
	if publication.Previous.Valid() {
		parents = append(parents, publication.Previous)
	}
	batch, err := artifact.NewDocumentBatch(
		"peer/capability/"+publication.ID.String(), []artifact.Content{capabilityContent, publicationContent},
		append(
			artifact.DependencyLineage(capability.ID, capability.Environment),
			artifact.DependencyLineage(publication.ID, parents...)...,
		), []artifact.AliasBinding{alias},
	)
	if err != nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return PeerCapabilityPublication{}, RemotePeerCapability{}, err
	}
	return publication, capability, nil
}

func canonicalizePeerCapabilityPublication(value *PeerCapabilityPublication) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Peer.Kind() != artifact.KindEvidence ||
		value.Capability.Kind() != artifact.KindEvidence || value.PublishedUnixNS <= 0 ||
		value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
		return errors.New("model recipe: invalid peer capability publication")
	}
	return nil
}
