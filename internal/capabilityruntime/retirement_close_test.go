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

// yieldingContext holds its request at the director's cancellation check, once the first model's
// Close has been let through, until the other request has loaded: the test, not the scheduler, says
// which of two requests takes the freed place.
type yieldingContext struct {
	context.Context
	armed, after <-chan struct{}
}

// Err waits only once armed, so until then the request evicts or parks as it would under a plain
// context, and it gives the wait up when the test ends.
func (c yieldingContext) Err() error {
	select {
	case <-c.armed:
		select {
		case <-c.after:
		case <-c.Done():
		}
	default:
	}
	return c.Context.Err()
}

// admission is how one contending request ended: with the lease it was given, or with why it got none.
type admission struct {
	lease *SessionLease[*gatedModel]
	err   error
}

// TestBlockedCloseHoldsCapacityOnEveryRetirementPath holds a director's capacity to what is still
// resident: a model leaves it only when its Close returns, whether it was evicted for another model
// or retired when its request ended, so at capacity one a Close that has not returned keeps a second
// model from loading and shows as retiring in the snapshot. Two requests contend for the place an
// eviction frees and nothing in the director says which takes it, so the test admits each of them
// first in turn and gives every lease back before it reads the next: in neither order does it wait on
// a request that is waiting on it.
func TestBlockedCloseHoldsCapacityOnEveryRetirementPath(t *testing.T) {
	t.Parallel()
	for name, row := range map[string]struct {
		session recipe.SessionPolicy
		// ahead names the idle-eviction request that takes the freed place: the other is held at its
		// cancellation check until that one has loaded. Empty where only one request contends.
		ahead string
	}{
		"idle eviction, the evicting request admitted first": {recipe.SessionCapacity, "second"},
		"idle eviction, the waiting request admitted first":  {recipe.SessionCapacity, "third"},
		"request retirement": {session: recipe.SessionRequest},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			ctx := t.Context()
			director, err := NewComponentSessionDirector[*gatedModel]("test", "host", 1)
			if err != nil {
				t.Fatal(err)
			}
			component := func(label string) modelrecipe.ComponentSession {
				return modelrecipe.ComponentSession{
					Identity: testutil.ArtifactID(t, artifact.KindProfile, label+"-resources"),
					Node:     "stage", Module: "test.stage",
					Model: testutil.ArtifactID(t, artifact.KindModel, label), Session: row.session,
				}
			}
			first := newGatedModel()
			held, err := director.LeaseComponent(ctx, component("first"), func(context.Context) (*gatedModel, error) { return first, nil })
			if err != nil {
				t.Fatal(err)
			}
			second, third := newGatedModel(), newGatedModel()
			close(second.gate)
			close(third.gate)
			secondLoaded, thirdLoaded := make(chan struct{}), make(chan struct{})
			secondCtx, thirdCtx, order := ctx, ctx, []*gatedModel{third}
			switch row.ahead {
			case "second":
				thirdCtx, order = yieldingContext{Context: ctx, armed: first.gate, after: secondLoaded}, []*gatedModel{second, third}
			case "third":
				secondCtx, order = yieldingContext{Context: ctx, armed: first.gate, after: thirdLoaded}, []*gatedModel{third, second}
			}
			// A contending request only reports how it ended, and the test's own goroutine judges that: a
			// request can end after the test has, when it may no longer fail it.
			admitted := make(chan admission, len(order))
			contend := func(ctx context.Context, wanted modelrecipe.ComponentSession, model *gatedModel, loaded chan struct{}) {
				lease, err := director.LeaseComponent(ctx, wanted, func(context.Context) (*gatedModel, error) {
					close(loaded)
					return model, nil
				})
				admitted <- admission{lease, err}
			}
			// The first model's teardown starts: an idle model is evicted by the next request for another, a
			// request-scoped one is retired when its lease ends.
			if row.session == recipe.SessionCapacity {
				if err := held.Release(); err != nil {
					t.Fatal(err)
				}
				go contend(secondCtx, component("second"), second, secondLoaded)
			} else {
				go func() { _ = held.Release() }()
			}
			<-first.closing
			if snapshot := director.Snapshot(); snapshot.Retiring != 1 {
				t.Fatalf("a model whose Close has not returned reads %d retiring, want 1: %+v", snapshot.Retiring, snapshot)
			}
			// A request that needs another model waits for the Close to return; it does not load beside it.
			go contend(thirdCtx, component("third"), third, thirdLoaded)
			for director.Snapshot().Waiting == 0 {
				select {
				case <-thirdLoaded:
					t.Fatal("a model loaded while another's Close had not returned")
				default:
					runtime.Gosched()
				}
			}
			close(first.gate)
			// At capacity one the request admitted first holds the only place, and the other is admitted
			// when that lease is back: so each lease is released before the next is read, whichever came.
			for _, want := range order {
				got := <-admitted
				if got.err != nil || got.lease == nil {
					t.Fatalf("a request that waited for the Close to return was not admitted: %v", got.err)
				}
				if got.lease.Model() != want {
					t.Fatal("the requests were admitted in the other order than this row sets out to force")
				}
				if err := got.lease.Release(); err != nil {
					t.Fatal(err)
				}
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
