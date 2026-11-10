package capabilityruntime

import (
	"context"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// countedModel counts the Closes of the models its counter is shared by.
type countedModel struct{ closed *atomic.Int32 }

func (m countedModel) Close(context.Context) error {
	m.closed.Add(1)
	return nil
}

// TestIdleSessionsGiveTheirMemoryBack holds what a model about to open asks of the process's media sessions
// (judge-speed/media-residency): every session no lease holds is closed and counted an eviction, one in use is left
// as it is, a released session loads again on its next request, and a closed director is no longer asked. It does
// not run in parallel, since the release reaches every director of the process.
func TestIdleSessionsGiveTheirMemoryBack(t *testing.T) {
	ctx := t.Context()
	director := testcheck.Must(NewComponentSessionDirector[countedModel]("test", "host", 2))(t)
	component := func(label string) modelrecipe.ComponentSession {
		return modelrecipe.ComponentSession{
			Identity: testutil.ArtifactID(t, artifact.KindProfile, label+"-resources"),
			Node:     "stage", Module: "test.stage",
			Model: testutil.ArtifactID(t, artifact.KindModel, label), Session: recipe.SessionCapacity,
		}
	}
	var closed atomic.Int32
	loads := 0
	load := func(context.Context) (countedModel, error) {
		loads++
		return countedModel{&closed}, nil
	}
	idle := testcheck.Must(director.LeaseComponent(ctx, component("idle"), load))(t)
	testcheck.NoError(t, idle.Release())
	busy := testcheck.Must(director.LeaseComponent(ctx, component("busy"), load))(t)
	released := testcheck.Must(ReleaseIdleSessions(ctx))(t)
	if snapshot := director.Snapshot(); released < 1 || closed.Load() != 1 || snapshot.Evictions != 1 || snapshot.Retiring != 0 {
		t.Fatalf("the release closed %d sessions in all, %d of this director's, with %+v; want the idle one closed and counted an eviction, the one in use left", released, closed.Load(), snapshot)
	}
	again := testcheck.Must(director.LeaseComponent(ctx, component("idle"), load))(t)
	if loads != 3 {
		t.Fatalf("the released session's next request made %d loads in all, want it loaded again, the third", loads)
	}
	testcheck.NoError(t, again.Release())
	testcheck.NoError(t, busy.Release())
	testcheck.NoError(t, director.Close(ctx))
	idleSessions.Lock()
	_, held := idleSessions.directors[director]
	idleSessions.Unlock()
	if held {
		t.Fatal("a closed director is still asked to release its sessions")
	}
}
