// Package capabilityruntime validates remote orchestration authority at its runtime boundary.
package capabilityruntime

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestOrchestrationAuthorityRefusals(t *testing.T) {
	peer := newPeerLifecycleFixture(t, "orchestration-peer")
	defer peer.store.Close()
	now := peerObservedUnixNS + 1
	authority, err := peer.authority.Resolve(context.Background(), peer.enrollment.ID, now)
	if err != nil {
		t.Fatal(err)
	}
	backend := &peerReconcileBackend{}
	manager, err := operation.NewManager(peerReconcileRetention)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	reconciler, err := (PeerReplicaReconcilerConfig{
		Repository: peer.store, Operations: manager, Backend: backend,
	}).Open()
	if err != nil {
		t.Fatal(err)
	}
	replica := peerReplicaFixture(t, peer.enrollment.ID, peer.enrollment.Environment, "orchestration-remote")
	replica.Endpoint = authority.Capability.Endpoint
	replica.Publication = authority.Publication.ID
	replica.Heartbeat = authority.Heartbeat.ID
	replica.Compatibility = testutil.ArtifactID(t, artifact.KindEvidence, "orchestration-compatibility")
	plan := peerReconcilePlan(t, replica)
	request := peerReconcileRequest(plan, nil)
	request.ChangedUnixNS = now
	id, err := reconciler.Reconcile(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(context.Background(), id)
	if err != nil || status.State != operation.StateCompleted || len(status.Outputs) != 2 {
		t.Fatalf("remote whole-model reconciliation=(%+v, %v)", status, err)
	}
	for _, phase := range []runrecord.PeerReplicaPhase{runrecord.PeerReplicaStage, runrecord.PeerReplicaLoad} {
		target, err := peerReplicaTarget(replica)
		if err != nil {
			t.Fatal(err)
		}
		receipt, found, err := runrecord.ResolvePeerReplicaReceipt(context.Background(), peer.store, id, target, phase)
		if err != nil || !found || receipt.Outcome != runrecord.PeerReplicaSucceeded || receipt.Plan != plan.Identity {
			t.Fatalf("remote %s evidence=(%+v, %t, %v)", phase, receipt, found, err)
		}
	}

	mismatched := replica
	mismatched.Publication = testutil.ArtifactID(t, artifact.KindEvidence, "stale-peer-publication")
	staleRequest := peerReconcileRequest(peerReconcilePlan(t, mismatched), nil)
	staleRequest.ChangedUnixNS = now
	if _, err := reconciler.Reconcile(context.Background(), staleRequest); err == nil ||
		!strings.Contains(err.Error(), "differs from current peer authority") {
		t.Fatalf("stale remote authority accepted: %v", err)
	}
	if _, err := peer.authority.Transition(
		context.Background(), peer.enrollment.ID, runrecord.PeerDraining, now+1,
	); err != nil {
		t.Fatal(err)
	}
	draining := peerReconcileRequest(plan, nil)
	draining.ChangedUnixNS = now + 2
	if _, err := reconciler.Reconcile(context.Background(), draining); err == nil ||
		!strings.Contains(err.Error(), "remote replica authority is unavailable") {
		t.Fatalf("draining remote authority accepted: %v", err)
	}
}
