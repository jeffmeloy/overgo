package capabilityruntime

import (
	"context"
	"runtime"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// gatedModel is a model whose Close announces itself and then waits for its gate, so a test holds a
// teardown open while it looks at what the director admits beside it.
type gatedModel struct {
	closing chan struct{}
	gate    chan struct{}
	// announce closes closing, once however often Close is called.
	announce func()
}

func newGatedModel() *gatedModel {
	closing := make(chan struct{})
	return &gatedModel{closing: closing, gate: make(chan struct{}), announce: sync.OnceFunc(func() { close(closing) })}
}

func (m *gatedModel) Close(context.Context) error {
	m.announce()
	<-m.gate
	return nil
}

// TestBlockedCloseHoldsCapacityOnEveryRetirementPath holds a director's capacity to what is still
// resident: a model leaves it only when its Close returns, whether it was evicted for another model
// or retired when its request ended, so at capacity one a Close that has not returned keeps a second
// model from loading and shows as retiring in the snapshot.
func TestBlockedCloseHoldsCapacityOnEveryRetirementPath(t *testing.T) {
	t.Parallel()
	for name, session := range map[string]recipe.SessionPolicy{"idle eviction": recipe.SessionCapacity, "request retirement": recipe.SessionRequest} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			ctx := t.Context()
			director, err := NewComponentSessionDirector[*gatedModel]("test", "host", 1)
			if err != nil {
				t.Fatal(err)
			}
			component := func(label string, policy recipe.SessionPolicy) modelrecipe.ComponentSession {
				return modelrecipe.ComponentSession{
					Identity: testutil.ArtifactID(t, artifact.KindProfile, label+"-resources"),
					Node:     "stage", Module: "test.stage",
					Model: testutil.ArtifactID(t, artifact.KindModel, label), Session: policy,
				}
			}
			first := newGatedModel()
			held, err := director.LeaseComponent(ctx, component("first", session), func(context.Context) (*gatedModel, error) { return first, nil })
			if err != nil {
				t.Fatal(err)
			}
			// The first model's teardown starts: an idle model is evicted by the next request for another, a
			// request-scoped one is retired when its lease ends.
			second := newGatedModel()
			close(second.gate)
			secondLoaded := make(chan *SessionLease[*gatedModel], 1)
			if session == recipe.SessionCapacity {
				if err := held.Release(); err != nil {
					t.Fatal(err)
				}
				go func() {
					lease, err := director.LeaseComponent(ctx, component("second", session), func(context.Context) (*gatedModel, error) { return second, nil })
					if err != nil {
						t.Error(err)
					}
					secondLoaded <- lease
				}()
			} else {
				go func() { _ = held.Release() }()
			}
			<-first.closing
			if snapshot := director.Snapshot(); snapshot.Retiring != 1 {
				t.Fatalf("a model whose Close has not returned reads %d retiring, want 1: %+v", snapshot.Retiring, snapshot)
			}
			// A request that needs another model waits for the Close to return; it does not load beside it.
			third := newGatedModel()
			close(third.gate)
			loaded := make(chan struct{})
			result := make(chan *SessionLease[*gatedModel], 1)
			go func() {
				lease, err := director.LeaseComponent(ctx, component("third", session), func(context.Context) (*gatedModel, error) {
					close(loaded)
					return third, nil
				})
				if err != nil {
					t.Error(err)
				}
				result <- lease
			}()
			for director.Snapshot().Waiting == 0 {
				select {
				case <-loaded:
					t.Fatal("a model loaded while another's Close had not returned")
				default:
					runtime.Gosched()
				}
			}
			close(first.gate)
			if session == recipe.SessionCapacity {
				if lease := <-secondLoaded; lease != nil {
					if err := lease.Release(); err != nil {
						t.Fatal(err)
					}
				}
			}
			lease := <-result
			if lease == nil {
				t.Fatal("the waiting request was never admitted")
			}
			if err := lease.Release(); err != nil {
				t.Fatal(err)
			}
			if err := director.Close(ctx); err != nil {
				t.Fatal(err)
			}
			if snapshot := director.Snapshot(); snapshot.Retiring != 0 {
				t.Fatalf("a finished teardown still reads %d retiring: %+v", snapshot.Retiring, snapshot)
			}
		})
	}
}
