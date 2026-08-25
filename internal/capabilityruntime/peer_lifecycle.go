package capabilityruntime

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

// PeerServingLease is the exact enrolled, active, capability-bound lease admitted for serving.
type PeerServingLease = modelrecipe.PeerServingAuthority

// PeerLifecycleAuthority owns RepoDB mutations and serving admission for enrolled peers.
type PeerLifecycleAuthority struct {
	Repository artifact.Repository
}

// PeerPlacementAuthority compiles deterministic whole-model replica placement.
type PeerPlacementAuthority struct {
	Repository artifact.Repository
}

// Compile resolves current recipe, locality, compatibility, resources, and peer leases.
func (authority PeerPlacementAuthority) Compile(
	ctx context.Context,
	request modelrecipe.PeerPlacementRequest,
) (modelrecipe.PeerPlacementPlan, error) {
	return modelrecipe.CompilePeerPlacementPlan(ctx, authority.Repository, request)
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
	return modelrecipe.ResolvePeerServingAuthority(ctx, authority.Repository, peer, nowUnixNS)
}
