package capabilityruntime

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

// PeerServingLease is the exact enrolled, active, capability-bound lease admitted for serving.
type PeerServingLease struct {
	Enrollment  runrecord.PeerEnrollment              `json:"enrollment"`
	State       runrecord.PeerState                   `json:"state"`
	Publication modelrecipe.PeerCapabilityPublication `json:"publication"`
	Capability  modelrecipe.RemotePeerCapability      `json:"capability"`
	Heartbeat   runrecord.PeerHeartbeat               `json:"heartbeat"`
}

// PeerLifecycleAuthority owns RepoDB mutations and serving admission for enrolled peers.
type PeerLifecycleAuthority struct {
	Repository artifact.Repository
}

// Enroll commits an approved public identity and its initial active state.
func (authority PeerLifecycleAuthority) Enroll(
	ctx context.Context,
	value runrecord.PeerEnrollment,
) (runrecord.PeerEnrollment, runrecord.PeerState, error) {
	return runrecord.PublishPeerEnrollment(ctx, authority.Repository, value)
}

// PublishCapability advances the active peer's declared serving transport.
func (authority PeerLifecycleAuthority) PublishCapability(
	ctx context.Context,
	peer artifact.ID,
	value modelrecipe.RemotePeerCapability,
	publishedUnixNS int64,
) (modelrecipe.PeerCapabilityPublication, modelrecipe.RemotePeerCapability, error) {
	return modelrecipe.PublishPeerCapability(ctx, authority.Repository, peer, value, publishedUnixNS)
}

// Heartbeat advances the peer's exact capability-bound lease.
func (authority PeerLifecycleAuthority) Heartbeat(
	ctx context.Context,
	value runrecord.PeerHeartbeat,
) (runrecord.PeerHeartbeat, error) {
	publication, _, found, err := modelrecipe.ResolvePeerCapabilityPublication(ctx, authority.Repository, value.Peer)
	if err != nil || !found || publication.ID != value.Capability {
		return runrecord.PeerHeartbeat{}, errors.Join(errors.New("capability runtime: heartbeat capability differs from peer authority"), err)
	}
	return runrecord.PublishPeerHeartbeat(ctx, authority.Repository, value)
}

// Transition advances the peer's one-way administrative lifecycle.
func (authority PeerLifecycleAuthority) Transition(
	ctx context.Context,
	peer artifact.ID,
	state runrecord.PeerAdministrativeState,
	changedUnixNS int64,
) (runrecord.PeerState, error) {
	return runrecord.PublishPeerState(ctx, authority.Repository, peer, state, changedUnixNS)
}

// Resolve requires an active peer and an unexpired heartbeat over its current capability.
func (authority PeerLifecycleAuthority) Resolve(
	ctx context.Context,
	peer artifact.ID,
	nowUnixNS int64,
) (PeerServingLease, error) {
	if ctx == nil || authority.Repository == nil || nowUnixNS <= 0 {
		return PeerServingLease{}, errors.New("capability runtime: invalid peer lease request")
	}
	enrollment, err := runrecord.RequirePeerEnrollment(ctx, authority.Repository, peer)
	if err != nil {
		return PeerServingLease{}, err
	}
	state, found, err := runrecord.ResolvePeerState(ctx, authority.Repository, peer)
	if err != nil || !found || state.State != runrecord.PeerActive {
		return PeerServingLease{}, errors.Join(errors.New("capability runtime: peer is not active"), err)
	}
	publication, capability, found, err := modelrecipe.ResolvePeerCapabilityPublication(ctx, authority.Repository, peer)
	if err != nil || !found || capability.Environment != enrollment.Environment {
		return PeerServingLease{}, errors.Join(errors.New("capability runtime: peer capability is unavailable"), err)
	}
	heartbeat, found, err := runrecord.ResolvePeerHeartbeat(ctx, authority.Repository, peer)
	if err != nil || !found || heartbeat.Peer != peer || heartbeat.Capability != publication.ID ||
		heartbeat.ObservedUnixNS > nowUnixNS || nowUnixNS >= heartbeat.ExpiresUnixNS {
		return PeerServingLease{}, errors.Join(errors.New("capability runtime: peer lease is absent, mismatched, or expired"), err)
	}
	return PeerServingLease{
		Enrollment: enrollment, State: state, Publication: publication,
		Capability: capability, Heartbeat: heartbeat,
	}, nil
}
