package capabilityruntime

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const (
	peerApprovedUnixNS  = int64(1_700_000_000_000_000_000)
	peerPublishedUnixNS = peerApprovedUnixNS + 10
	peerObservedUnixNS  = peerPublishedUnixNS + 10
	peerExpiresUnixNS   = peerObservedUnixNS + 100
)

type peerLifecycleFixture struct {
	authority   PeerLifecycleAuthority
	store       *overgodb.Store
	enrollment  runrecord.PeerEnrollment
	publication modelrecipe.PeerCapabilityPublication
}

func TestPeerLifecycleAuthority(t *testing.T) {
	fixture := newPeerLifecycleFixture(t, "peer-lifecycle")
	defer fixture.store.Close()
	lease, err := fixture.authority.Resolve(t.Context(), fixture.enrollment.ID, peerObservedUnixNS+1)
	if err != nil || lease.Enrollment.ID != fixture.enrollment.ID || lease.State.State != runrecord.PeerActive ||
		lease.Publication.ID != fixture.publication.ID || lease.Heartbeat.Capability != fixture.publication.ID ||
		lease.Capability.Environment != fixture.enrollment.Environment {
		t.Fatalf("peer lease=(%+v, %v)", lease, err)
	}
}

func TestPeerIdentityBinding(t *testing.T) {
	fixture := newPeerLifecycleFixture(t, "peer-identity")
	defer fixture.store.Close()
	other := enrollPeerFixture(t, fixture.authority, fixture.enrollment.Environment, "other-peer", peerApprovedUnixNS+1)
	if _, err := fixture.authority.Heartbeat(t.Context(), runrecord.PeerHeartbeat{
		Peer: other.ID, Capability: fixture.publication.ID,
		ObservedUnixNS: peerObservedUnixNS + 1, ExpiresUnixNS: peerExpiresUnixNS + 1,
	}); err == nil {
		t.Fatal("capability publication from another peer was accepted")
	}
	otherEnvironment := testutil.ArtifactID(t, artifact.KindEvidence, "other-peer-environment")
	if _, _, err := fixture.authority.PublishCapability(t.Context(), fixture.enrollment.ID, modelrecipe.RemotePeerCapability{
		Environment: otherEnvironment, Tasks: []recipe.Task{recipe.TaskGeneration}, Endpoint: "https://peer.example/v1/generation",
	}, peerPublishedUnixNS+1); err == nil {
		t.Fatal("capability for another environment was accepted")
	}
}

func TestPeerLeaseExpiry(t *testing.T) {
	fixture := newPeerLifecycleFixture(t, "peer-expiry")
	defer fixture.store.Close()
	if _, err := fixture.authority.Resolve(t.Context(), fixture.enrollment.ID, peerExpiresUnixNS); err == nil {
		t.Fatal("expired peer lease was admitted")
	}
	if _, err := fixture.authority.Resolve(t.Context(), fixture.enrollment.ID, peerObservedUnixNS-1); err == nil {
		t.Fatal("future peer heartbeat was admitted")
	}
}

func TestPeerDrainRefusal(t *testing.T) {
	fixture := newPeerLifecycleFixture(t, "peer-drain")
	defer fixture.store.Close()
	if _, err := fixture.authority.Transition(
		t.Context(), fixture.enrollment.ID, runrecord.PeerDraining, peerObservedUnixNS+1,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.authority.Resolve(t.Context(), fixture.enrollment.ID, peerObservedUnixNS+2); err == nil {
		t.Fatal("draining peer admitted new serving work")
	}
	if _, _, err := fixture.authority.PublishCapability(t.Context(), fixture.enrollment.ID, modelrecipe.RemotePeerCapability{
		Environment: fixture.enrollment.Environment, Tasks: []recipe.Task{recipe.TaskInference},
		Endpoint: "https://peer.example/v1/inference",
	}, peerObservedUnixNS+2); err == nil {
		t.Fatal("draining peer changed capability publication")
	}
	if _, err := fixture.authority.Transition(
		t.Context(), fixture.enrollment.ID, runrecord.PeerRetired, peerObservedUnixNS+2,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.authority.Heartbeat(t.Context(), runrecord.PeerHeartbeat{
		Peer: fixture.enrollment.ID, Capability: fixture.publication.ID,
		ObservedUnixNS: peerObservedUnixNS + 3, ExpiresUnixNS: peerExpiresUnixNS + 3,
	}); err == nil {
		t.Fatal("retired peer heartbeat was accepted")
	}
}

func newPeerLifecycleFixture(t *testing.T, name string) peerLifecycleFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	environment := testutil.ArtifactID(t, artifact.KindEvidence, name+"-environment")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "peer/fixture/environment/" + name, Artifacts: []artifact.Descriptor{{ID: environment}},
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	authority := PeerLifecycleAuthority{Repository: store}
	enrollment := enrollPeerFixture(t, authority, environment, name, peerApprovedUnixNS)
	publication, _, err := authority.PublishCapability(t.Context(), enrollment.ID, modelrecipe.RemotePeerCapability{
		Environment: environment, Tasks: []recipe.Task{recipe.TaskGeneration}, Endpoint: "https://peer.example/v1/generation",
	}, peerPublishedUnixNS)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if _, err := authority.Heartbeat(t.Context(), runrecord.PeerHeartbeat{
		Peer: enrollment.ID, Capability: publication.ID,
		ObservedUnixNS: peerObservedUnixNS, ExpiresUnixNS: peerExpiresUnixNS,
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	return peerLifecycleFixture{authority: authority, store: store, enrollment: enrollment, publication: publication}
}

func enrollPeerFixture(
	t *testing.T,
	authority PeerLifecycleAuthority,
	environment artifact.ID,
	name string,
	approvedUnixNS int64,
) runrecord.PeerEnrollment {
	t.Helper()
	public, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	enrollment, state, err := authority.Enroll(t.Context(), runrecord.PeerEnrollment{
		Name: name, Environment: environment, PublicKey: public,
		ApprovedBy: "test-operator", ApprovedUnixNS: approvedUnixNS,
	})
	if err != nil || state.State != runrecord.PeerActive || state.Peer != enrollment.ID {
		t.Fatalf("peer enrollment=(%+v, %+v, %v)", enrollment, state, err)
	}
	return enrollment
}
