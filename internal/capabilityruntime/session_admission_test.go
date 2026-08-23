package capabilityruntime

import (
	"context"
	"errors"
	"runtime"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestBoundedSessionAdmission(t *testing.T) {
	t.Run("queue bound and cancellation", func(t *testing.T) {
		director, err := NewResidentModelSessionDirector("test", "host", 1, struct{}{})
		if err != nil {
			t.Fatal(err)
		}
		defer director.Close(context.Background())
		held, ok := director.TryLease(-1)
		if !ok {
			t.Fatal("initial lease was not admitted")
		}
		ctx, cancel := context.WithCancel(context.Background())
		parked := make(chan error, 1)
		go func() {
			_, err := director.Lease(ctx, -1)
			parked <- err
		}()
		waitForParkedSession(t, director)
		if _, err := director.Lease(context.Background(), -1); !errors.Is(err, ErrAdmissionQueueFull) {
			t.Fatalf("overflow error = %v", err)
		}
		cancel()
		if err := <-parked; !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled waiter error = %v", err)
		}
		if err := held.Release(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("requested slot and shutdown", func(t *testing.T) {
		director, err := NewResidentModelSessionDirector("test", "host", 2, struct{}{})
		if err != nil {
			t.Fatal(err)
		}
		held, ok := director.TryLease(1)
		if !ok {
			t.Fatal("requested slot lease was not admitted")
		}
		parked := make(chan *SessionLease[struct{}], 1)
		parkedErr := make(chan error, 1)
		go func() {
			lease, err := director.Lease(context.Background(), 1)
			parked <- lease
			parkedErr <- err
		}()
		waitForParkedSession(t, director)
		other, ok := director.TryLease(0)
		if !ok {
			t.Fatal("unrequested slot was not independently available")
		}
		if err := held.Release(); err != nil {
			t.Fatal(err)
		}
		resumed := <-parked
		if err := <-parkedErr; err != nil || resumed == nil || resumed.ID != 1 {
			t.Fatalf("resumed requested lease = (%v, %v)", resumed, err)
		}
		shutdownCtx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := director.Lease(shutdownCtx, -1); !errors.Is(err, context.Canceled) {
			t.Fatalf("pre-cancelled lease error = %v", err)
		}
		if err := other.Release(); err != nil {
			t.Fatal(err)
		}
		if err := resumed.Release(); err != nil {
			t.Fatal(err)
		}
		if err := director.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
		if _, err := director.Lease(context.Background(), -1); !errors.Is(err, ErrSessionUnavailable) {
			t.Fatalf("closed director error = %v", err)
		}
	})

	t.Run("shutdown wakes parked request", func(t *testing.T) {
		director, err := NewResidentModelSessionDirector("test", "host", 1, struct{}{})
		if err != nil {
			t.Fatal(err)
		}
		held, ok := director.TryLease(-1)
		if !ok {
			t.Fatal("initial lease was not admitted")
		}
		parked := make(chan error, 1)
		go func() {
			_, err := director.Lease(context.Background(), -1)
			parked <- err
		}()
		waitForParkedSession(t, director)
		closed := make(chan error, 1)
		go func() { closed <- director.Close(context.Background()) }()
		if err := <-parked; !errors.Is(err, ErrSessionUnavailable) {
			t.Fatalf("shutdown waiter error = %v", err)
		}
		if err := held.Release(); err != nil {
			t.Fatal(err)
		}
		if err := <-closed; err != nil {
			t.Fatal(err)
		}
	})
}

type lifecycleSessionModel struct{ closes *int }

func (m *lifecycleSessionModel) Close(context.Context) error {
	*m.closes++
	return nil
}

func TestSessionDirectorBoundsComponentWaitingAndReportsLifecycle(t *testing.T) {
	closes := 0
	director, err := NewComponentSessionDirector[*lifecycleSessionModel]("test", "host", 1)
	if err != nil {
		t.Fatal(err)
	}
	component := func(name string) modelrecipe.ComponentSession {
		return modelrecipe.ComponentSession{
			Identity: testutil.ArtifactID(t, artifact.KindProfile, name+"-resources"),
			Node:     "stage", Module: "test.stage",
			Model: testutil.ArtifactID(t, artifact.KindModel, name), Session: recipe.SessionCapacity,
		}
	}
	first := component("first")
	held, err := director.LeaseComponent(t.Context(), first, func(context.Context) (*lifecycleSessionModel, error) {
		return &lifecycleSessionModel{closes: &closes}, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	cancelledContext, cancel := context.WithCancel(t.Context())
	cancelled := make(chan error, 1)
	go func() {
		_, leaseErr := director.LeaseComponent(cancelledContext, first, func(context.Context) (*lifecycleSessionModel, error) {
			return nil, errors.New("same component was loaded while resident")
		})
		cancelled <- leaseErr
	}()
	waitForParkedSession(t, director)
	cancel()
	if err := <-cancelled; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled component waiter error = %v", err)
	}

	resumed := make(chan *SessionLease[*lifecycleSessionModel], 1)
	resumedErr := make(chan error, 1)
	go func() {
		lease, leaseErr := director.LeaseComponent(t.Context(), first, func(context.Context) (*lifecycleSessionModel, error) {
			return nil, errors.New("same component was reloaded")
		})
		resumed <- lease
		resumedErr <- leaseErr
	}()
	waitForParkedSession(t, director)
	if _, err := director.LeaseComponent(t.Context(), first, func(context.Context) (*lifecycleSessionModel, error) {
		return nil, errors.New("overflow waiter loaded")
	}); !errors.Is(err, ErrAdmissionQueueFull) {
		t.Fatalf("component overflow error = %v", err)
	}
	if err := held.Release(); err != nil {
		t.Fatal(err)
	}
	reused := <-resumed
	if err := <-resumedErr; err != nil || reused == nil {
		t.Fatalf("resumed component lease = (%v, %v)", reused, err)
	}
	snapshot := director.Snapshot()
	if snapshot.Active != 1 || snapshot.Waiting != 0 || snapshot.Entries != 1 ||
		snapshot.Loads != 1 || snapshot.Reuses != 1 || snapshot.Evictions != 0 {
		t.Fatalf("reused lifecycle snapshot = %+v", snapshot)
	}
	if err := reused.Release(); err != nil {
		t.Fatal(err)
	}

	second := component("second")
	replacement, err := director.LeaseComponent(t.Context(), second, func(context.Context) (*lifecycleSessionModel, error) {
		return &lifecycleSessionModel{closes: &closes}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	snapshot = director.Snapshot()
	if snapshot.Active != 1 || snapshot.Available != 0 || snapshot.Entries != 1 ||
		snapshot.Loads != 2 || snapshot.Reuses != 1 || snapshot.Evictions != 1 || closes != 1 {
		t.Fatalf("evicted lifecycle snapshot = %+v closes=%d", snapshot, closes)
	}
	if err := replacement.Release(); err != nil {
		t.Fatal(err)
	}
	snapshot = director.Snapshot()
	if snapshot.Active != 0 || snapshot.Available != 1 || snapshot.Idle != 1 {
		t.Fatalf("idle lifecycle snapshot = %+v", snapshot)
	}
	if err := director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if closes != 2 {
		t.Fatalf("closed models = %d", closes)
	}
}

func TestSessionAuthorityPublishesReplacementBeforeBorrowerDrain(t *testing.T) {
	closes := 0
	director, err := NewComponentSessionDirector[*lifecycleSessionModel]("test", "host", 2)
	if err != nil {
		t.Fatal(err)
	}
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, name)
	}
	scope := id(artifact.KindProfile, "replacement-scope")
	firstKey := sessionKey{
		model:     id(artifact.KindModel, "replacement-first-model"),
		resources: id(artifact.KindProfile, "replacement-first-resources"), device: "host", policy: string(recipe.SessionCapacity),
	}
	secondKey := sessionKey{
		model:     id(artifact.KindModel, "replacement-second-model"),
		resources: id(artifact.KindProfile, "replacement-second-resources"), device: "host", policy: string(recipe.SessionCapacity),
	}
	load := func(context.Context) (*lifecycleSessionModel, error) {
		return &lifecycleSessionModel{closes: &closes}, nil
	}
	first, fresh, err := director.leaseEntry(
		t.Context(), firstKey, scope, nil, load,
	)
	if err != nil || !fresh {
		t.Fatalf("first lease = %v, fresh=%t", err, fresh)
	}
	second, fresh, err := director.leaseEntry(
		t.Context(), secondKey, scope, nil, load,
	)
	if err != nil || !fresh {
		t.Fatalf("replacement lease = %v, fresh=%t", err, fresh)
	}
	if snapshot := director.Snapshot(); snapshot.Entries != 1 || snapshot.Retiring != 1 || snapshot.Retirements != 1 {
		t.Fatalf("replacement snapshot = %+v", snapshot)
	}
	director.release(first)
	if snapshot := director.Snapshot(); snapshot.Retiring != 0 || closes != 1 {
		t.Fatalf("drained snapshot = %+v closes=%d", snapshot, closes)
	}
	director.release(second)
	if err := director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if closes != 2 {
		t.Fatalf("closed models = %d", closes)
	}
}

func TestSessionAuthorityRebindsCompatibleExecution(t *testing.T) {
	director, err := NewComponentSessionDirector[*lifecycleSessionModel]("test", "host", 1)
	if err != nil {
		t.Fatal(err)
	}
	closes, loads := 0, 0
	key := sessionKey{
		model:     testutil.ArtifactID(t, artifact.KindModel, "rebind-model"),
		resources: testutil.ArtifactID(t, artifact.KindProfile, "rebind-resources"),
		device:    "host", policy: string(recipe.SessionCapacity),
	}
	lease := func(scope artifact.ID) (*sessionEntry[*lifecycleSessionModel], bool, error) {
		return director.leaseEntry(t.Context(), key, scope, nil, func(context.Context) (*lifecycleSessionModel, error) {
			loads++
			return &lifecycleSessionModel{closes: &closes}, nil
		})
	}
	first, fresh, err := lease(testutil.ArtifactID(t, artifact.KindProfile, "rebind-first-scope"))
	if err != nil || !fresh {
		t.Fatalf("first lease = %v, fresh=%t", err, fresh)
	}
	director.release(first)
	secondScope := testutil.ArtifactID(t, artifact.KindProfile, "rebind-second-scope")
	second, fresh, err := lease(secondScope)
	if err != nil || fresh || loads != 1 || second.scope != secondScope {
		t.Fatalf("rebound lease = %v, fresh=%t loads=%d scope=%s", err, fresh, loads, second.scope)
	}
	director.release(second)
	if err := director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if closes != 1 {
		t.Fatalf("closed models = %d", closes)
	}
}

func waitForParkedSession[Input, Model, Output any](t *testing.T, director *ModelSessionDirector[Input, Model, Output]) {
	t.Helper()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()
	for director.Snapshot().Waiting == 0 {
		select {
		case <-timer.C:
			t.Fatal("session did not enter admission parking")
		default:
			runtime.Gosched()
		}
	}
}
