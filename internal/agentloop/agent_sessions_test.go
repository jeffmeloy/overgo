package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// A generated session's shape: its seeds and the operations of each.
const (
	agentSessionSeeds = 40
	agentSessionSteps = 30
)

// How the fake tool answers a call: with its result, with a failure, or by cancelling the call while it is in flight.
const (
	agentToolAnswers int32 = iota
	agentToolFails
	agentToolCancels
)

// agentReference is what a person may expect of a session, operation by operation: its steps, whether its next step has
// an inspection, and whether a live grant binds the next write.
type agentReference struct {
	steps     int
	inspected bool
	granted   bool
}

// agentWorld is one session's coordinator and store, and the fake tools behind it: how they answer, the call to cancel
// in flight, and how many times the write ran.
type agentWorld struct {
	coordinator *Coordinator
	store       *overgodb.Store
	mode        atomic.Int32
	cancel      atomic.Value
	writes      atomic.Int64
}

func newAgentWorld(t *testing.T) *agentWorld {
	t.Helper()
	ctx := t.Context()
	store := overgodbtest.Create(t, t.TempDir())
	world := &agentWorld{store: store}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "agent-session-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "agent-session-model")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "agent-session/identity", Artifacts: []artifact.Descriptor{{ID: recipeID}}}); err != nil {
		t.Fatal(err)
	}
	answer := func(w http.ResponseWriter, r *http.Request) {
		switch world.mode.Load() {
		case agentToolFails:
			http.Error(w, "tool error", http.StatusInternalServerError)
		case agentToolCancels:
			_, _ = io.Copy(io.Discard, r.Body)
			world.cancel.Load().(context.CancelFunc)()
			<-r.Context().Done()
		default:
			_, _ = w.Write([]byte(`{"ok":true}`))
		}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /read", answer)
	mux.HandleFunc("POST /write", func(w http.ResponseWriter, r *http.Request) {
		world.writes.Add(1)
		answer(w, r)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	inspect, err := agenttool.NewManual(agenttool.Manual{
		Name: "session.read", Description: "Read state for the session.", Effect: agenttool.EffectInspection,
		Arguments: []agenttool.Field{{Name: "step", Kind: agenttool.FieldInteger}},
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/read"},
	})
	testcheck.NoError(t, err)
	mutate, err := agenttool.NewManual(agenttool.Manual{
		Name: "session.write", Description: "Change state for the session.", Effect: agenttool.EffectMutation,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/write"},
	})
	testcheck.NoError(t, err)
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{inspect, mutate}); err != nil {
		t.Fatal(err)
	}
	coordinator, err := New(store, agenttool.NewOperatorExecutor(), Identity{Recipe: recipeID, Model: modelID, Node: recipe.NodeID("respond")}, 1000)
	testcheck.NoError(t, err)
	world.coordinator = coordinator
	return world
}

// inFlight is a call context the tool cancels while the call is in flight.
func (w *agentWorld) inFlight(ctx context.Context) (context.Context, context.CancelFunc) {
	callCtx, cancel := context.WithCancel(ctx)
	w.cancel.Store(cancel)
	w.mode.Store(agentToolCancels)
	return callCtx, cancel
}

// agentSession drives seed's session and says what broke the reference first, with the operations before it.
func agentSession(t *testing.T, seed uint64) (string, []string) {
	world := newAgentWorld(t)
	ctx := t.Context()
	rng := rand.New(rand.NewPCG(seed, seed))
	session := &Session{ID: fmt.Sprintf("generated-%d", seed)}
	var reference agentReference
	var trace []string
	reads := 0
	empty := json.RawMessage(`{}`)
	read := func(callCtx context.Context) error {
		reads++
		_, err := world.coordinator.Propose(callCtx, session, "session.read", json.RawMessage(fmt.Sprintf(`{"step":%d}`, reads)))
		return err
	}
	for range agentSessionSteps {
		preview, err := world.coordinator.PreviewMutationDecision(ctx, session, "session.write", empty)
		switch {
		case err != nil:
			return fmt.Sprintf("the preview refuses: %v", err), trace
		case preview.Binds != reference.granted:
			return fmt.Sprintf("the preview binds %t, the reference %t", preview.Binds, reference.granted), trace
		case session.Steps != reference.steps:
			return fmt.Sprintf("%d steps, the reference %d", session.Steps, reference.steps), trace
		case session.Inspection.Valid() != reference.inspected:
			return fmt.Sprintf("inspected %t, the reference %t", session.Inspection.Valid(), reference.inspected), trace
		}
		switch rng.IntN(10) {
		case 0, 1:
			trace = append(trace, "read")
			if err := read(ctx); err != nil {
				return fmt.Sprintf("a read refuses: %v", err), trace
			}
			reference.steps++
			reference.inspected, reference.granted = true, false
		case 2:
			trace = append(trace, "read fails")
			world.mode.Store(agentToolFails)
			err := read(ctx)
			world.mode.Store(agentToolAnswers)
			if err == nil {
				return "a broken read succeeded", trace
			}
			reference.granted = false
		case 3:
			trace = append(trace, "read cancelled")
			callCtx, cancel := world.inFlight(ctx)
			err := read(callCtx)
			cancel()
			world.mode.Store(agentToolAnswers)
			if err == nil {
				return "a cancelled read succeeded", trace
			}
			reference.granted = false
		case 4, 5:
			trace = append(trace, "approve")
			_, err := world.coordinator.ApproveMutation(ctx, session, "session.write", empty, preview.Operation)
			if !reference.inspected && err == nil {
				return "an approval with no inspection was taken", trace
			}
			if reference.inspected && err != nil {
				return fmt.Sprintf("an approval after an inspection refuses: %v", err), trace
			}
			reference.granted = reference.inspected
		case 6, 7:
			trace = append(trace, "propose")
			before := world.writes.Load()
			_, err := world.coordinator.Propose(ctx, session, "session.write", empty)
			ran := world.writes.Load() - before
			switch {
			case reference.granted && err != nil:
				return fmt.Sprintf("a granted write refuses: %v", err), trace
			case reference.granted && ran != 1:
				return fmt.Sprintf("a granted write ran the tool %d times", ran), trace
			case !reference.granted && err == nil:
				return "a write with no live grant ran", trace
			case !reference.granted && ran != 0:
				return "a refused write ran the tool", trace
			}
			if reference.granted {
				reference.steps++
				reference.inspected, reference.granted = false, false
			}
		case 8:
			mode := agentToolFails + int32(rng.IntN(2))
			trace = append(trace, map[int32]string{agentToolFails: "propose fails", agentToolCancels: "propose cancelled"}[mode])
			before := world.writes.Load()
			var err error
			if mode == agentToolCancels {
				callCtx, cancel := world.inFlight(ctx)
				_, err = world.coordinator.Propose(callCtx, session, "session.write", empty)
				cancel()
			} else {
				world.mode.Store(agentToolFails)
				_, err = world.coordinator.Propose(ctx, session, "session.write", empty)
			}
			world.mode.Store(agentToolAnswers)
			ran := world.writes.Load() - before
			if err == nil {
				return "a broken write succeeded", trace
			}
			if ran > 1 || (ran == 1) != reference.granted {
				return fmt.Sprintf("a broken write ran the tool %d times with a grant %t", ran, reference.granted), trace
			}
			reference.granted = false
			// The step's receipt is closed once the call has returned: failed or completed, never admitted or running.
			tip, found, err := runrecord.ResolveStageReceipt(ctx, world.store, preview.Operation, recipe.NodeID("respond"))
			if err != nil {
				return fmt.Sprintf("the receipt: %v", err), trace
			}
			if found && tip.State != runrecord.StageFailed && tip.State != runrecord.StageCompleted {
				return fmt.Sprintf("a broken write left its receipt %s", tip.State), trace
			}
		default:
			trace = append(trace, "restore")
			restored, err := world.coordinator.RestoreSession(ctx, session.ID)
			if err != nil {
				return fmt.Sprintf("a restore refuses: %v", err), trace
			}
			session = restored
		}
	}
	return "", trace
}

// TestGeneratedAgentSessions drives agentSessionSeeds generated sessions of reads, approvals and writes, each plain,
// broken by the tool or cancelled in flight, with a restore at any point, and holds every session, operation by
// operation, to a small reference of its steps, its inspection and its grant: a granted write runs the tool once, a
// write with no live grant is refused and runs nothing, and a broken write leaves its receipt failed or completed.
func TestGeneratedAgentSessions(t *testing.T) {
	t.Parallel()
	broken := 0
	for seed := range uint64(agentSessionSeeds) {
		if failure, trace := agentSession(t, seed+1); failure != "" {
			broken++
			t.Errorf("seed %d: %s after %s", seed+1, failure, strings.Join(trace, "; "))
		}
	}
	if broken != 0 {
		t.Errorf("%d of %d sessions broke the reference", broken, agentSessionSeeds)
	}
}
