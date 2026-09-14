package gate

import (
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestGateAdmissionStoreReuse(t *testing.T) {
	repo, storePath := newLifecycleRepo(t), "store"
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate-admission@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Admission Test")
	runGitFixture(t, repo, "commit", "--allow-empty", "-q", "-m", "baseline")
	t.Setenv("OVERGO_STRATEGY_ID", "")

	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := requireNoPendingGateStateWithStore(repo, store); err != nil {
		t.Fatal(err)
	}
	g := gateContext{
		repo: repo, storePath: storePath, start: time.Unix(200, 0),
		environment: lifecycleTestEnvironment(t),
	}
	if err := g.prepareWithStore(store); err != nil {
		t.Fatal(err)
	}
	if !g.preparationCommit.Valid() {
		t.Fatal("reused admission store published no preparation commit")
	}
}

func TestGateStoreAcceleration(t *testing.T) {
	t.Parallel()
	repo, storePath := newLifecycleRepo(t), "store"
	store, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte("checkpoint acceleration fixture"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/gate-store-acceleration", Artifacts: []artifact.Descriptor{{ID: id}},
	}); err != nil {
		t.Fatal(err)
	}
	if accelerated, err := ensureGateStoreAcceleration(t.Context(), store); err != nil || !accelerated {
		t.Fatalf("cold store acceleration = (%t, %v)", accelerated, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := overgodb.Open(filepath.Join(repo, storePath))
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if replay := reopened.SnapshotReplay(); !replay.Loaded || replay.Fallback != "" {
		t.Fatalf("accelerated replay = %+v", replay)
	}
	if accelerated, err := ensureGateStoreAcceleration(t.Context(), reopened); err != nil || accelerated {
		t.Fatalf("checkpointed store acceleration = (%t, %v)", accelerated, err)
	}
}
