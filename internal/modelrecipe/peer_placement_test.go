package modelrecipe

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const (
	placementApprovedUnixNS  = int64(1_710_000_000_000_000_000)
	placementPublishedUnixNS = placementApprovedUnixNS + 10
	placementObservedUnixNS  = placementPublishedUnixNS + 10
	placementExpiresUnixNS   = placementObservedUnixNS + 100
	placementCompileUnixNS   = placementObservedUnixNS + 1
	placementEndpoint        = "https://placement-peer.example/v1/generation"
)

type peerPlacementFixture struct {
	capabilitySelectorFixture
	peer             artifact.ID
	compatibility    artifact.ID
	localObservation artifact.ID
	remoteLocation   artifact.Location
}

func TestCompilePeerPlacementPlan(t *testing.T) {
	fixture := newPeerPlacementFixture(t, true)
	request := fixture.request()
	first, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request)
	if err != nil || second.Identity != first.Identity || len(first.Replicas) != 2 ||
		first.Replicas[0].Peer != fixture.peer || first.Replicas[1].Peer.Valid() ||
		len(first.Components) != 1 || first.Components[0].Placement != recipe.PlacementHost || first.Components[0].Residency != "" {
		t.Fatalf("placement plans=(%+v, %+v, %v)", first, second, err)
	}
}

func TestPeerPlacementRefusalReasons(t *testing.T) {
	fixture := newPeerPlacementFixture(t, true)
	request := fixture.request()
	request.Policy.AllowLocal = false
	request.Policy.MinimumReplicas = 1
	request.Policy.MaximumReplicas = 1
	request.Policy.TargetConcurrency = 1
	request.Policy.MaximumMeasuredNS = 1
	if _, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request); err == nil ||
		!strings.Contains(err.Error(), "resources") || !strings.Contains(err.Error(), "latency") {
		t.Fatalf("resource refusal=%v", err)
	}
	request.Policy.MaximumMeasuredNS = 0
	request.Peers[0].Compatibility = testutil.ArtifactID(t, artifact.KindEvidence, "wrong-compatibility")
	if _, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request); err == nil ||
		!strings.Contains(err.Error(), "compatibility") {
		t.Fatalf("compatibility refusal=%v", err)
	}
}

func TestPeerPlacementArtifactLocality(t *testing.T) {
	fixture := newPeerPlacementFixture(t, false)
	request := fixture.request()
	request.Policy.AllowLocal = false
	request.Policy.MinimumReplicas = 1
	request.Policy.MaximumReplicas = 1
	request.Policy.TargetConcurrency = 1
	if _, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request); err == nil ||
		!strings.Contains(err.Error(), "artifact-locality") {
		t.Fatalf("missing locality refusal=%v", err)
	}
	if _, err := fixture.store.Commit(context.Background(), artifact.Batch{
		Key: "placement/remote-locality", Locations: []artifact.LocationEvent{{
			Location: fixture.remoteLocation, Action: artifact.LocationAdd,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	plan, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request)
	if err != nil || len(plan.Replicas) != 1 || plan.Replicas[0].Peer != fixture.peer || len(plan.Replicas[0].Locality) != 1 {
		t.Fatalf("remote locality plan=(%+v, %v)", plan, err)
	}
}

func TestPeerReplicaPolicyDerivation(t *testing.T) {
	fixture := newPeerPlacementFixture(t, true)
	request := fixture.request()
	request.Policy.TargetConcurrency = 3
	request.Policy.ConcurrencyPerReplica = 2
	plan, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request)
	if err != nil || len(plan.Replicas) != 2 {
		t.Fatalf("derived replicas=(%d, %v)", len(plan.Replicas), err)
	}
	request.Policy.MaximumReplicas = 1
	if _, err := CompilePeerPlacementPlan(context.Background(), fixture.store, request); err == nil ||
		!strings.Contains(err.Error(), "exceeds replica policy") {
		t.Fatalf("underprovisioned service policy accepted: %v", err)
	}
}

func newPeerPlacementFixture(t *testing.T, publishRemoteLocation bool) peerPlacementFixture {
	t.Helper()
	ctx := context.Background()
	fixture := newCapabilitySelectorFixture(t)
	localEnvironment := testutil.ArtifactID(t, artifact.KindEvidence, "placement-local-environment")
	peerEnvironment := testutil.ArtifactID(t, artifact.KindEvidence, "placement-peer-environment")
	if _, err := fixture.store.Commit(ctx, artifact.Batch{
		Key: "placement/environments", Artifacts: []artifact.Descriptor{{ID: localEnvironment}, {ID: peerEnvironment}},
	}); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(modelPath, []byte("placement-model"), 0o600); err != nil {
		t.Fatal(err)
	}
	localLocation, err := artifact.CanonicalLocalLocation(fixture.model, artifact.LocationFile, modelPath)
	if err != nil {
		t.Fatal(err)
	}
	remoteLocation := artifact.Location{Artifact: fixture.model, Kind: artifact.LocationRemote, Value: placementEndpoint}
	locations := []artifact.LocationEvent{{Location: localLocation, Action: artifact.LocationAdd}}
	if publishRemoteLocation {
		locations = append(locations, artifact.LocationEvent{Location: remoteLocation, Action: artifact.LocationAdd})
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{Key: "placement/locality", Locations: locations}); err != nil {
		t.Fatal(err)
	}
	observe := func(environment artifact.ID, measured, peak uint64) runrecord.ServingObservation {
		observation, err := runrecord.PublishServingObservation(ctx, fixture.store, runrecord.ServingObservation{
			Model: fixture.model, Recipe: fixture.definition.ID, Environment: environment, Task: recipe.TaskGeneration,
			Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: placementApprovedUnixNS,
			MeasuredNS: measured, Resources: runrecord.ServingResources{PeakDeviceBytes: peak},
		})
		if err != nil {
			t.Fatal(err)
		}
		return observation
	}
	localObservation := observe(localEnvironment, 20, 40)
	peerObservation := observe(peerEnvironment, 10, 30)
	selection, err := ResolveActiveExecution(ctx, fixture.store, fixture.model, recipe.TaskGeneration, SessionWarm)
	if err != nil {
		t.Fatal(err)
	}
	resourceContent, err := selection.Resources.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.Commit(ctx, artifact.Batch{Key: "placement/resources", Contents: []artifact.Content{resourceContent}}); err != nil {
		t.Fatal(err)
	}
	public, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	enrollment, _, err := runrecord.PublishPeerEnrollment(ctx, fixture.store, runrecord.PeerEnrollment{
		Name: "placement-peer", Environment: peerEnvironment, PublicKey: public,
		ApprovedBy: "placement-operator", ApprovedUnixNS: placementApprovedUnixNS,
	})
	if err != nil {
		t.Fatal(err)
	}
	capability := RemotePeerCapability{
		Environment: peerEnvironment, Tasks: []recipe.Task{recipe.TaskGeneration}, Endpoint: placementEndpoint,
	}
	publication, publishedCapability, err := PublishPeerCapability(
		ctx, fixture.store, enrollment.ID, capability, placementPublishedUnixNS,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.PublishPeerHeartbeat(ctx, fixture.store, runrecord.PeerHeartbeat{
		Peer: enrollment.ID, Capability: publication.ID,
		ObservedUnixNS: placementObservedUnixNS, ExpiresUnixNS: placementExpiresUnixNS,
	}); err != nil {
		t.Fatal(err)
	}
	compatibility, err := PublishRemotePeerAuthority(ctx, fixture.store, "placement/compatibility", capability, RemotePeerCompatibility{
		Model: fixture.model, Recipe: fixture.definition.ID, Resources: selection.Resources.Identity, Task: recipe.TaskGeneration,
		LocalEnvironment: localEnvironment, PeerEnvironment: peerEnvironment, PeerCapability: publishedCapability.ID,
		LocalObservation: localObservation.ID, PeerObservation: peerObservation.ID,
	})
	if err != nil {
		t.Fatal(err)
	}
	return peerPlacementFixture{
		capabilitySelectorFixture: fixture, peer: enrollment.ID, compatibility: compatibility.ID,
		localObservation: localObservation.ID, remoteLocation: remoteLocation,
	}
}

func (fixture peerPlacementFixture) request() PeerPlacementRequest {
	return PeerPlacementRequest{
		Model: fixture.model, Task: recipe.TaskGeneration, NowUnixNS: placementCompileUnixNS,
		LocalObservation: fixture.localObservation,
		Policy: PeerReplicaPolicy{
			MinimumReplicas: 1, MaximumReplicas: 2, TargetConcurrency: 2, ConcurrencyPerReplica: 1,
			MaximumMeasuredNS: 100, MaximumDeviceBytes: 100, AllowLocal: true, AllowPeers: true,
		},
		Peers: []PeerPlacementCandidate{{Peer: fixture.peer, Compatibility: fixture.compatibility}},
	}
}
