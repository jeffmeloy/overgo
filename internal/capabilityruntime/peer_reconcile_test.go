package capabilityruntime

import (
	"context"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

const (
	peerReconcileRetention = 8
	peerReconcileAttempts  = uint32(2)
	peerReconcileChangedNS = int64(1_720_000_000_000_000_000)
)

type peerBackendCall struct {
	phase   runrecord.PeerReplicaPhase
	replica modelrecipe.PeerReplicaPlacement
}

type peerReconcileBackend struct {
	mu              sync.Mutex
	calls           []peerBackendCall
	activeLeases    int
	loadStarted     chan struct{}
	cancelFirstLoad bool
	loads           int
}

func (backend *peerReconcileBackend) Stage(_ context.Context, replica modelrecipe.PeerReplicaPlacement) error {
	backend.record(runrecord.PeerReplicaStage, replica)
	return nil
}

func (backend *peerReconcileBackend) Load(ctx context.Context, replica modelrecipe.PeerReplicaPlacement) error {
	backend.mu.Lock()
	backend.loads++
	load := backend.loads
	backend.calls = append(backend.calls, peerBackendCall{phase: runrecord.PeerReplicaLoad, replica: replica})
	started, cancelFirst := backend.loadStarted, backend.cancelFirstLoad
	backend.mu.Unlock()
	if cancelFirst && load == 1 {
		close(started)
		<-ctx.Done()
		return ctx.Err()
	}
	return nil
}

func (backend *peerReconcileBackend) ActiveLeases(_ context.Context, _ modelrecipe.PeerReplicaPlacement) (int, error) {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	return backend.activeLeases, nil
}

func (backend *peerReconcileBackend) Unload(_ context.Context, replica modelrecipe.PeerReplicaPlacement) error {
	backend.record(runrecord.PeerReplicaUnload, replica)
	return nil
}

func (backend *peerReconcileBackend) record(phase runrecord.PeerReplicaPhase, replica modelrecipe.PeerReplicaPlacement) {
	backend.mu.Lock()
	backend.calls = append(backend.calls, peerBackendCall{phase: phase, replica: replica})
	backend.mu.Unlock()
}

func (backend *peerReconcileBackend) snapshot() []peerBackendCall {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	return append([]peerBackendCall(nil), backend.calls...)
}

func TestPeerReplicaReconciliation(t *testing.T) {
	fixture := newPeerReconcileFixture(t, &peerReconcileBackend{})
	replica := fixture.localReplica(t, "reconcile")
	status := fixture.run(t, peerReconcilePlan(t, replica), nil)
	if status.State != operation.StateCompleted || len(status.Outputs) != 2 {
		t.Fatalf("reconciliation status=%+v", status)
	}
	calls := fixture.backend.snapshot()
	if len(calls) != 2 || calls[0].phase != runrecord.PeerReplicaStage || calls[1].phase != runrecord.PeerReplicaLoad {
		t.Fatalf("reconciliation calls=%+v", calls)
	}
}

func TestPeerArtifactStaging(t *testing.T) {
	fixture := newPeerReconcileFixture(t, &peerReconcileBackend{})
	replica := fixture.localReplica(t, "staging")
	status := fixture.run(t, peerReconcilePlan(t, replica), nil)
	calls := fixture.backend.snapshot()
	if status.State != operation.StateCompleted || len(calls) != 2 ||
		calls[0].phase != runrecord.PeerReplicaStage || calls[0].replica.Locality[0] != replica.Locality[0] ||
		calls[1].phase != runrecord.PeerReplicaLoad {
		t.Fatalf("staging status=%+v calls=%+v", status, calls)
	}
}

func TestPeerZeroLeaseUnload(t *testing.T) {
	backend := &peerReconcileBackend{activeLeases: 1}
	fixture := newPeerReconcileFixture(t, backend)
	previous := fixture.localReplica(t, "leased")
	current := fixture.localReplica(t, "replacement")
	status := fixture.run(t, peerReconcilePlan(t, current), []modelrecipe.PeerReplicaPlacement{previous})
	if status.State != operation.StateBlocked || status.Recovery == nil || len(backend.snapshot()) != 0 {
		t.Fatalf("leased unload status=%+v calls=%+v", status, backend.snapshot())
	}
}

func TestPeerDrainCompletion(t *testing.T) {
	peer := newPeerLifecycleFixture(t, "reconcile-drain")
	defer peer.store.Close()
	if _, err := peer.authority.Transition(
		context.Background(), peer.enrollment.ID, runrecord.PeerDraining, peerReconcileChangedNS-1,
	); err != nil {
		t.Fatal(err)
	}
	manager, err := operation.NewManager(peerReconcileRetention)
	if err != nil {
		t.Fatal(err)
	}
	defer manager.Close()
	backend := &peerReconcileBackend{}
	reconciler, err := (PeerReplicaReconcilerConfig{Repository: peer.store, Operations: manager, Backend: backend}).Open()
	if err != nil {
		t.Fatal(err)
	}
	current := peerReplicaFixture(t, artifact.ID{}, peer.enrollment.Environment, "drain-current")
	previous := peerReplicaFixture(t, peer.enrollment.ID, peer.enrollment.Environment, "drain-previous")
	status := runPeerReconciliation(t, reconciler, manager, peerReconcilePlan(t, current), []modelrecipe.PeerReplicaPlacement{previous})
	state, found, err := runrecord.ResolvePeerState(context.Background(), peer.store, peer.enrollment.ID)
	if err != nil || !found || status.State != operation.StateCompleted || state.State != runrecord.PeerRetired {
		t.Fatalf("drain status=%+v state=%+v found=%t err=%v", status, state, found, err)
	}
}

func TestPeerReconcileRecovery(t *testing.T) {
	backend := &peerReconcileBackend{loadStarted: make(chan struct{}), cancelFirstLoad: true}
	fixture := newPeerReconcileFixture(t, backend)
	plan := peerReconcilePlan(t, fixture.localReplica(t, "recovery"))
	ctx, cancel := context.WithCancel(context.Background())
	id, err := fixture.reconciler.Reconcile(ctx, peerReconcileRequest(plan, nil))
	if err != nil {
		t.Fatal(err)
	}
	<-backend.loadStarted
	cancel()
	status, err := fixture.manager.Wait(context.Background(), id)
	if err != nil || status.State != operation.StateCancelled {
		t.Fatalf("cancelled reconciliation=%+v err=%v", status, err)
	}
	fixture.manager.Close()
	restarted, err := operation.NewManager(peerReconcileRetention)
	if err != nil {
		t.Fatal(err)
	}
	defer restarted.Close()
	reconciler, err := (PeerReplicaReconcilerConfig{
		Repository: fixture.store, Operations: restarted, Backend: backend,
	}).Open()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reconciler.Recover(context.Background(), id, peerReconcileRequest(plan, nil)); err != nil {
		t.Fatal(err)
	}
	status, err = restarted.Wait(context.Background(), id)
	calls := backend.snapshot()
	if err != nil || status.State != operation.StateCompleted || len(calls) != 3 ||
		calls[0].phase != runrecord.PeerReplicaStage || calls[1].phase != runrecord.PeerReplicaLoad ||
		calls[2].phase != runrecord.PeerReplicaLoad {
		t.Fatalf("recovered reconciliation=%+v calls=%+v err=%v", status, calls, err)
	}
}

type peerReconcileFixture struct {
	store      *overgodb.Store
	manager    *operation.Manager
	reconciler *PeerReplicaReconciler
	backend    *peerReconcileBackend
}

func newPeerReconcileFixture(t *testing.T, backend *peerReconcileBackend) peerReconcileFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	manager, err := operation.NewManager(peerReconcileRetention)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(manager.Close)
	reconciler, err := (PeerReplicaReconcilerConfig{Repository: store, Operations: manager, Backend: backend}).Open()
	if err != nil {
		t.Fatal(err)
	}
	return peerReconcileFixture{store: store, manager: manager, reconciler: reconciler, backend: backend}
}

func (fixture peerReconcileFixture) localReplica(t *testing.T, name string) modelrecipe.PeerReplicaPlacement {
	t.Helper()
	return peerReplicaFixture(t, artifact.ID{}, testutil.ArtifactID(t, artifact.KindEvidence, name+"-environment"), name)
}

func (fixture peerReconcileFixture) run(
	t *testing.T,
	plan modelrecipe.PeerPlacementPlan,
	previous []modelrecipe.PeerReplicaPlacement,
) operation.Status {
	t.Helper()
	return runPeerReconciliation(t, fixture.reconciler, fixture.manager, plan, previous)
}

func runPeerReconciliation(
	t *testing.T,
	reconciler *PeerReplicaReconciler,
	manager *operation.Manager,
	plan modelrecipe.PeerPlacementPlan,
	previous []modelrecipe.PeerReplicaPlacement,
) operation.Status {
	t.Helper()
	id, err := reconciler.Reconcile(context.Background(), peerReconcileRequest(plan, previous))
	if err != nil {
		t.Fatal(err)
	}
	status, err := manager.Wait(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return status
}

func peerReconcileRequest(
	plan modelrecipe.PeerPlacementPlan,
	previous []modelrecipe.PeerReplicaPlacement,
) PeerReconcileRequest {
	return PeerReconcileRequest{
		Plan: plan, Previous: previous, Policy: PeerReconcilePolicy{MaximumAttempts: peerReconcileAttempts},
		ChangedUnixNS: peerReconcileChangedNS,
	}
}

func peerReplicaFixture(
	t *testing.T,
	peer artifact.ID,
	environment artifact.ID,
	name string,
) modelrecipe.PeerReplicaPlacement {
	t.Helper()
	model := testutil.ArtifactID(t, artifact.KindModel, name+"-model")
	return modelrecipe.PeerReplicaPlacement{
		Peer: peer, Environment: environment, Observation: testutil.ArtifactID(t, artifact.KindEvidence, name+"-observation"),
		Resources: runrecord.ServingResources{}, Locality: []artifact.Location{{
			Artifact: model, Kind: artifact.LocationRemote, Value: "https://peer.example/" + name,
		}},
	}
}

func peerReconcilePlan(t *testing.T, replicas ...modelrecipe.PeerReplicaPlacement) modelrecipe.PeerPlacementPlan {
	t.Helper()
	for index := range replicas {
		replicas[index].Index = index
	}
	selection := testutil.ArtifactID(t, artifact.KindProfile, "reconcile-selection")
	model := testutil.ArtifactID(t, artifact.KindModel, "reconcile-model")
	definition := testutil.ArtifactID(t, artifact.KindRecipe, "reconcile-recipe")
	resources := testutil.ArtifactID(t, artifact.KindProfile, "reconcile-resources")
	policy := testutil.ArtifactID(t, artifact.KindProfile, "reconcile-policy")
	components := []modelrecipe.ComponentSession{{
		Node: "generate", Module: modelrecipe.ModuleThoughtBankGenerate, Model: model,
		Session: recipe.SessionCapacity, Placement: recipe.PlacementHost,
	}}
	identity, err := artifact.JSONID(artifact.KindProfile, struct {
		Selection  artifact.ID                        `json:"selection"`
		Model      artifact.ID                        `json:"model"`
		Task       recipe.Task                        `json:"task"`
		Recipe     artifact.ID                        `json:"recipe"`
		Resources  artifact.ID                        `json:"resources"`
		Policy     artifact.ID                        `json:"policy"`
		Components []modelrecipe.ComponentSession     `json:"components"`
		Replicas   []modelrecipe.PeerReplicaPlacement `json:"replicas"`
	}{selection, model, recipe.TaskGeneration, definition, resources, policy, components, replicas})
	if err != nil {
		t.Fatal(err)
	}
	plan := modelrecipe.PeerPlacementPlan{
		Identity: identity, Selection: selection, Model: model, Task: recipe.TaskGeneration,
		Recipe: definition, Resources: resources, Policy: policy, Components: components, Replicas: replicas,
	}
	if err := plan.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	return plan
}
