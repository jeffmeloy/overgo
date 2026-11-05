package agentloop

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

const attemptTarget = "state.txt"

var noArguments = json.RawMessage(`{}`)

// attemptTools publishes an inspection and a mutation of one workspace file over a server of the test's own.
func attemptTools(t *testing.T, store *overgodb.Store, handle http.HandlerFunc) {
	t.Helper()
	server := httptest.NewServer(handle)
	t.Cleanup(server.Close)
	ceiling := agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeWorkspace, Value: attemptTarget}}}
	look, err := agenttool.NewManual(agenttool.Manual{
		Name: "attempt.look", Description: "Read the state file.", Effect: agenttool.EffectInspection, Ceiling: ceiling,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/look"},
	})
	testcheck.NoError(t, err)
	change, err := agenttool.NewManual(agenttool.Manual{
		Name: "attempt.change", Description: "Change the state file.", Effect: agenttool.EffectMutation, Ceiling: ceiling,
		Transport: agenttool.Transport{Kind: agenttool.TransportHTTP, URL: server.URL + "/change"},
	})
	testcheck.NoError(t, err)
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, []agenttool.Manual{look, change}); err != nil {
		t.Fatal(err)
	}
}

// approveChange grants the session's next attempt, as an operator does from the preview.
func approveChange(t *testing.T, coordinator *Coordinator, session *Session) {
	t.Helper()
	if _, err := coordinator.ApproveMutation(t.Context(), session, "attempt.change", noArguments,
		previewedOperation(t, coordinator, session, "attempt.change")); err != nil {
		t.Fatal(err)
	}
}

// TestAToolErrorLeavesTheSessionUsable: a failed tool takes no step, so the step is tried again under its next ordinal. An
// inspection runs again at once. A mutation needs a new approval, since a grant authorizes one execution, and its checkpoint
// names the attempt that ran. An approval outside the agent's manuals takes no ordinal.
func TestAToolErrorLeavesTheSessionUsable(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	var failing atomic.Bool
	attemptTools(t, store, func(w http.ResponseWriter, _ *http.Request) {
		if failing.Swap(false) {
			http.Error(w, "broken", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(`{"done":true}`))
	})
	root := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, attemptTarget), []byte("state"), 0o600))
	checkpoints, err := NewMutationCheckpointRuntime(root, store)
	testcheck.NoError(t, err)
	session := &Session{ID: "attempts", Checkpoints: checkpoints}

	failing.Store(true)
	if _, err := coordinator.Propose(ctx, session, "attempt.look", noArguments); err == nil {
		t.Fatal("a broken inspection succeeded")
	}
	if _, err := coordinator.Propose(ctx, session, "attempt.look", noArguments); err != nil {
		t.Fatalf("an inspection could not be tried again: %v", err)
	}
	if session.Steps != 1 {
		t.Fatalf("steps = %d after one inspection that failed once", session.Steps)
	}

	preview, err := coordinator.PreviewMutationDecision(ctx, session, "attempt.change", noArguments)
	testcheck.NoError(t, err)
	refused, err := coordinator.PreviewMutationDecision(ctx, session, "attempt.look", noArguments)
	testcheck.NoError(t, err)
	agent := testutil.ArtifactID(t, artifact.KindRecipe, "attempt-agent")
	if _, err := coordinator.ApproveMutationWithManuals(ctx, session, "attempt.change", noArguments, preview.Operation, agent, []artifact.ID{refused.Manual}); err == nil {
		t.Fatal("an approval outside the agent's manuals was granted")
	}
	if _, found, err := runrecord.ResolveAttemptStimulus(ctx, store, preview.Operation, 1); err != nil || found {
		t.Fatalf("a refused approval took an ordinal: found=%t err=%v", found, err)
	}

	approveChange(t, coordinator, session)
	failing.Store(true)
	if _, err := coordinator.Propose(ctx, session, "attempt.change", noArguments); err == nil {
		t.Fatal("a broken mutation succeeded")
	}
	if _, err := coordinator.Propose(ctx, session, "attempt.change", noArguments); err == nil || !strings.Contains(err.Error(), "approve it again") {
		t.Fatalf("a spent grant ran again: %v", err)
	}
	spent, err := coordinator.PreviewMutationDecision(ctx, session, "attempt.change", noArguments)
	if err != nil || spent.Decision == nil || spent.Binds {
		t.Fatalf("preview of a spent grant = %+v, %v", spent, err)
	}

	// Approved twice, so the stimulus that runs is the third and the receipt's attempt the second.
	approveChange(t, coordinator, session)
	approveChange(t, coordinator, session)
	if _, err := coordinator.Propose(ctx, session, "attempt.change", noArguments); err != nil {
		t.Fatalf("a mutation could not be tried again: %v", err)
	}
	if session.Steps != fixtureSessionSteps {
		t.Fatalf("steps = %d, want %d", session.Steps, fixtureSessionSteps)
	}
	tip, found, err := runrecord.ResolveStageReceipt(ctx, store, preview.Operation, coordinator.identity.Node)
	if err != nil || !found || tip.State != runrecord.StageCompleted || tip.Attempt != fixtureSessionSteps {
		t.Fatalf("receipt = %+v, found=%t, err=%v", tip, found, err)
	}
	interaction, found, err := runrecord.ResolveInteraction(ctx, store, preview.CallID)
	if err != nil || !found || interaction.Run != tip.ID || len(interaction.Tools) != 1 {
		t.Fatalf("interaction = %+v, found=%t, err=%v", interaction, found, err)
	}
	checkpoint, err := runrecord.RequireAgentMutationCheckpoint(ctx, store, interaction.Tools[0])
	testcheck.NoError(t, err)
	stimulus, found, err := runrecord.ResolveAttemptStimulus(ctx, store, preview.Operation, checkpoint.Attempt)
	if err != nil || !found || stimulus.ID != interaction.Stimulus || tip.Invocation.Preflight != stimulus.ID {
		t.Fatalf("checkpoint attempt %d, stimulus %+v, found=%t, err=%v", checkpoint.Attempt, stimulus, found, err)
	}
}

// TestACancelledMutationClosesItsReceipt: a request that goes away during a mutation leaves a closed receipt, never one
// marked running, and the step can be approved and tried again.
func TestACancelledMutationClosesItsReceipt(t *testing.T) {
	t.Parallel()
	coordinator, store := coordinatorFixture(t)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	var cancelling atomic.Bool
	attemptTools(t, store, func(w http.ResponseWriter, r *http.Request) {
		if cancelling.Swap(false) {
			io.Copy(io.Discard, r.Body) // the server notices a disconnect only past the body
			cancel()
			<-r.Context().Done()
			return
		}
		w.Write([]byte(`{"done":true}`))
	})
	session := &Session{ID: "cancelled"}
	if _, err := coordinator.Propose(ctx, session, "attempt.look", noArguments); err != nil {
		t.Fatal(err)
	}
	preview, err := coordinator.PreviewMutationDecision(ctx, session, "attempt.change", noArguments)
	testcheck.NoError(t, err)
	approveChange(t, coordinator, session)
	cancelling.Store(true)
	if _, err := coordinator.Propose(ctx, session, "attempt.change", noArguments); err == nil {
		t.Fatal("a cancelled mutation succeeded")
	}
	tip, found, err := runrecord.ResolveStageReceipt(t.Context(), store, preview.Operation, coordinator.identity.Node)
	if err != nil || !found || tip.State != runrecord.StageFailed {
		t.Fatalf("receipt after a cancelled request = %+v, found=%t, err=%v", tip, found, err)
	}
	approveChange(t, coordinator, session)
	if _, err := coordinator.Propose(t.Context(), session, "attempt.change", noArguments); err != nil {
		t.Fatalf("a cancelled mutation could not be tried again: %v", err)
	}
}

// TestASessionTipReadsWhatARestoreFinds: after an inspection the tip says inspected and after a mutation it does not, as a
// restored session does.
func TestASessionTipReadsWhatARestoreFinds(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	attemptTools(t, store, func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"done":true}`)) })
	session := &Session{ID: "tip"}
	agrees := func(step int) {
		t.Helper()
		tip, err := coordinator.ReadSessionTip(ctx, session.ID, step)
		testcheck.NoError(t, err)
		restored, err := coordinator.RestoreSession(ctx, session.ID)
		testcheck.NoError(t, err)
		if tip.Steps != restored.Steps || tip.Interaction != restored.Interaction || tip.Inspected != restored.Inspection.Valid() {
			t.Fatalf("tip %+v, restored %+v", tip, restored)
		}
	}
	if _, err := coordinator.Propose(ctx, session, "attempt.look", noArguments); err != nil {
		t.Fatal(err)
	}
	agrees(1)
	approveChange(t, coordinator, session)
	if _, err := coordinator.Propose(ctx, session, "attempt.change", noArguments); err != nil {
		t.Fatal(err)
	}
	agrees(fixtureSessionSteps)
}
