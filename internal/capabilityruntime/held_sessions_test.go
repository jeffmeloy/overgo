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

// TestAHeldRequestScopedSessionStaysLoaded holds a render's suite of one model to one load (HoldSessions): a
// request-scoped session whose request ends under the hold is kept, the next request of the model leases it without
// loading, and it closes with its director; without the hold it is closed as its request ends and the next request
// loads it again, as a server frees the card between requests.
func TestAHeldRequestScopedSessionStaysLoaded(t *testing.T) {
	t.Parallel()
	for _, held := range []bool{true, false} {
		ctx := t.Context()
		if held {
			ctx = HoldSessions(ctx)
		}
		director := testcheck.Must(NewComponentSessionDirector[countedModel]("test", "host", 1))(t)
		component := modelrecipe.ComponentSession{
			Identity: testutil.ArtifactID(t, artifact.KindProfile, "picture-resources"),
			Node:     "stage", Module: "test.stage",
			Model: testutil.ArtifactID(t, artifact.KindModel, "picture"), Session: recipe.SessionRequest,
		}
		var closed atomic.Int32
		loads := 0
		load := func(context.Context) (countedModel, error) {
			loads++
			return countedModel{&closed}, nil
		}
		for range 3 {
			lease := testcheck.Must(director.LeaseComponent(ctx, component, load))(t)
			testcheck.NoError(t, lease.Release())
		}
		wantLoads, wantClosed := 1, int32(0)
		if !held {
			wantLoads, wantClosed = 3, 3
		}
		if loads != wantLoads || closed.Load() != wantClosed {
			t.Errorf("held %t: three requests made %d loads and %d closes, want %d and %d", held, loads, closed.Load(), wantLoads, wantClosed)
		}
		testcheck.NoError(t, director.Close(t.Context()))
		if held && closed.Load() != 1 {
			t.Errorf("a held session closed %d times with its director, want once", closed.Load())
		}
	}
}
