package audioparity

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/adaptertrain"
	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/speechrecognition"
	"overgo/internal/testskip"
	"overgo/internal/trainingdata"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// cancelDecodeContext cancels at the first decoder-internal cancellation poll,
// after DecodeAudio's entry admission. No sleeps or CPU-speed assumptions enter
// the interruption witness; the underlying context supplies Done and Err.
type cancelDecodeContext struct {
	context.Context
	cancel  context.CancelCauseFunc
	entered bool
}

func (ctx *cancelDecodeContext) Err() error {
	if ctx.entered {
		ctx.cancel(context.Canceled)
	}
	ctx.entered = true
	return ctx.Context.Err()
}

func TestAudioCancellationAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": real audio cancellation runs as its exact mandatory batch acceptance")
	}
	l := newAdapterLifecycle(t)
	t.Run("decode", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(context.Canceled)
		probe := &cancelDecodeContext{Context: ctx, cancel: cancel}
		audio, _, err := media.DecodeAudio(probe, l.fixture.payload, l.fixture.record.Samples)
		if !errors.Is(err, context.Canceled) || !probe.entered || len(audio.Samples) != 0 {
			t.Fatalf("decoder returned partial output or lost cancellation: %v", err)
		}
	})
	t.Run("encoder-boundary", func(t *testing.T) {
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(context.Canceled)
		var workspace speechrecognition.Workspace
		observed := false
		hidden, frames, err := l.fixture.encoder.Encode(ctx, l.fixture.features, l.fixture.frames, &workspace, func(trace speechrecognition.Trace) error {
			if trace.Block >= 0 && len(trace.Values) != 0 {
				observed = true
				cancel(context.Canceled)
			}
			return nil
		})
		if !observed || !errors.Is(err, context.Canceled) || len(hidden) != 0 || frames != 0 {
			t.Fatalf("real encoder interruption: observed=%t err=%v", observed, err)
		}
		// A fresh invocation reuses the interrupted workspace without retaining
		// partial output as a valid transcript or changing frozen parameters.
		hidden, frames, err = l.fixture.encoder.Encode(t.Context(), l.fixture.features, l.fixture.frames, &workspace, nil)
		if err != nil || frames != l.frames || !slices.Equal(hidden, l.hidden) {
			t.Fatalf("encoder replay after cancellation: %v", err)
		}
	})
	t.Run("adapter-update-boundary", func(t *testing.T) {
		weights := l.adapter.WeightSnapshot()
		state, err := l.adapter.OptimizerSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(context.Canceled)
		execution, err := l.adapter.Bind(ctx)
		if err != nil {
			t.Fatal(err)
		}
		gradients, err := execution.Select(trainingprogram.PhaseForward, trainingprogram.PhaseBackward)
		if err != nil {
			t.Fatal(err)
		}
		update, err := execution.Select(trainingprogram.PhaseOptimize)
		if err != nil {
			t.Fatal(err)
		}
		example := adaptertrain.LinearCTCExample{Hidden: l.hidden, Frames: l.frames, Targets: l.fixture.targets}
		if err := gradients.Run(&example); err != nil || len(example.Gradient) == 0 {
			t.Fatalf("real adapter gradients: %v", err)
		}
		cancel(context.Canceled)
		if err := update.Run(&example); !errors.Is(err, context.Canceled) || !slices.Equal(weights, l.adapter.WeightSnapshot()) {
			t.Fatalf("cancelled update mutated parameters: %v", err)
		}
		if _, err := l.adapter.OptimizerSnapshot(); err == nil {
			t.Fatal("unfinished update advertised a checkpoint")
		}
		if err := l.adapter.Restore(weights, state); err != nil {
			t.Fatal(err)
		}
		actual, err := l.adapter.OptimizerSnapshot()
		if err != nil || !reflect.DeepEqual(actual, state) {
			t.Fatalf("checkpoint restore after cancelled update: %v", err)
		}
	})
	t.Run("dataset-cursor", func(t *testing.T) {
		stream, err := trainingdata.NewStream(l.data, nil)
		if err != nil {
			t.Fatal(err)
		}
		batcher, err := trainingdata.NewBatcher(stream, trainingdata.BatchPolicy{Examples: 1, DecodeWorkers: 1, MaxBytes: adapterAcceptanceMemory})
		if err != nil {
			t.Fatal(err)
		}
		before := stream.Snapshot()
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		if _, err := batcher.Next(ctx); !errors.Is(err, context.Canceled) || stream.Snapshot() != before {
			t.Fatalf("cancelled batch advanced cursor: %v", err)
		}
		batch, err := batcher.Next(t.Context())
		if err != nil || len(batch.Examples) != 1 || batch.Examples[0].ID != l.fixture.record.RowID {
			t.Fatalf("source replay after cancellation: %v", err)
		}
	})
	t.Run("session-release", func(t *testing.T) {
		session, err := speechrecognition.LoadSession(t.Context(), l.store, l.base.ID, adapterAcceptanceMemory)
		if err != nil {
			t.Fatal(err)
		}
		defer session.Close(context.WithoutCancel(t.Context()))
		lease, err := session.Lease(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		_, _, err = lease.Transcribe(ctx, l.fixture.payload, l.origin, l.policy, speechrecognition.RunBinding{Key: "cancelled/session"})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled transcription: %v", err)
		}
		if err := lease.Release(); err != nil {
			t.Fatal(err)
		}
		if snapshot := session.Snapshot(); snapshot.Active != 0 || snapshot.Waiting != 0 {
			t.Fatalf("cancelled lease retained admission: %+v", snapshot)
		}
		if err := session.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
		if _, err := session.Lease(t.Context()); err == nil {
			t.Fatal("closed model session admitted execution")
		}
	})
	t.Run("http-operation", func(t *testing.T) {
		commit := strings.TrimSpace(baselineCommand(t, l.fixture.root, "git", "rev-parse", "HEAD"))
		handler, runtime := newAudioHTTPFixture(t, l.store, l.base, l.policy, commit)
		entered := make(chan artifact.ID, 1)
		runtime.observe = func(ctx context.Context, reporter operation.Reporter) {
			entered <- reporter.OperationID()
			<-ctx.Done()
		}
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(context.Canceled)
		request := audioHTTPRequest(t, l.base, l.fixture.payload, true).WithContext(ctx)
		response := httptest.NewRecorder()
		done := make(chan struct{})
		go func() { defer close(done); handler.ServeHTTP(response, request) }()
		var id artifact.ID
		select {
		case id = <-entered:
		case <-done:
			t.Fatalf("HTTP operation did not enter the real workspace: %d", response.Code)
		case <-t.Context().Done():
			t.Fatal("HTTP operation admission did not finish")
		}
		cancel(context.Canceled)
		select {
		case <-done:
		case <-t.Context().Done():
			t.Fatal("cancelled HTTP request did not finish")
		}
		if err := handler.Close(); err != nil {
			t.Fatal(err)
		}
		statuses := audioHTTPOperations(t, handler)
		if response.Code == http.StatusOK || len(statuses) != 1 || statuses[0].ID != id || statuses[0].State != operation.StateCancelled || statuses[0].Run == nil {
			t.Fatalf("HTTP cancellation lacked terminal operation: %+v", statuses)
		}
		run, err := runrecord.RequireExactRun(t.Context(), l.store, *statuses[0].Run)
		if err != nil || run.Outcome != runrecord.OutcomeCancelled || run.Recipe != l.base.ID {
			t.Fatalf("HTTP cancellation record: %v", err)
		}
	})
	t.Log("real CPU FLAC decode and encoder-boundary cancellation, completed adapter gradients with cancelled update, restored optimizer, unchanged dataset cursor, released/closed model session and joined cancelled HTTP operation; no forced process termination or GPU claim")
}

func TestAudioPublicationRecoveryAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration + ": real audio publication recovery runs as its exact mandatory batch acceptance")
	}
	l := newAdapterLifecycle(t)
	step := l.update(t, l.batcher(t, nil))
	commit := strings.TrimSpace(baselineCommand(t, l.fixture.root, "git", "rev-parse", "HEAD"))
	handler, _ := newAudioHTTPFixture(t, l.store, l.base, l.policy, commit)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, audioHTTPRequest(t, l.base, l.fixture.payload, true))
	if response.Code != http.StatusOK {
		t.Fatalf("predecessor HTTP inference: %d", response.Code)
	}
	directory := filepath.Join(t.TempDir(), "complete")
	checkpoint := l.publish(t, directory, step.Stream)
	// The standard publisher owns staging and rename. Interrupt after the real
	// weight write but before checkpoint publication, then prove no partial target.
	spec := trainingprogram.CheckpointSpec{RunPlan: checkpoint.RunPlan, Program: checkpoint.Program, Model: checkpoint.Model, Dataset: checkpoint.Dataset, Split: checkpoint.Split,
		Stream: checkpoint.Stream, Optimizer: checkpoint.Optimizer, ParameterCount: checkpoint.ParameterCount, RNG: checkpoint.RNG, Processors: checkpoint.Processors, Lineage: checkpoint.Lineage}
	partial := filepath.Join(t.TempDir(), "interrupted")
	interrupted := errors.New("injected publication-boundary interruption")
	_, err := trainingprogram.PublishCheckpoint(partial, spec, func(stage string) error {
		if err := l.adapter.SaveWeights(stage, l.binding); err != nil {
			return err
		}
		return interrupted
	})
	if !errors.Is(err, interrupted) {
		t.Fatalf("publication interruption: %v", err)
	}
	if _, err := os.Stat(partial); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("partial checkpoint target visible")
	}
	if _, err := trainingprogram.LoadCheckpoint(partial); err == nil {
		t.Fatal("partial checkpoint loaded")
	}
	if complete, err := trainingprogram.LoadCheckpoint(directory); err != nil || complete.ID() != checkpoint.ID() {
		t.Fatal("predecessor checkpoint changed")
	}
	batch, err := checkpoint.Batch("recovery/checkpoint", directory)
	if err != nil {
		t.Fatal(err)
	}
	fault := &vadFaultRepository{Repository: l.store, fail: true}
	head, sequence := l.store.Head()
	if _, err := artifact.CommitBatch(t.Context(), fault, batch); !errors.Is(err, errVADPublication) {
		t.Fatalf("store publication interruption: %v", err)
	}
	if current, count := l.store.Head(); current != head || count != sequence {
		t.Fatal("failed checkpoint batch advanced store")
	}
	if _, found, err := l.store.Artifact(t.Context(), checkpoint.ID()); err != nil || found {
		t.Fatal("failed checkpoint batch published an artifact")
	}
	candidate, err := modelrecipe.AdaptedTranscriptionDefinition(l.base, checkpoint.ID())
	if err != nil {
		t.Fatal(err)
	}
	priorVerification, err := modelrecipetest.PublishVerification(t.Context(), l.store, "recovery/predecessor-verification", l.base.ID)
	if err != nil {
		t.Fatal(err)
	}
	refuseActivation := func() {
		t.Helper()
		if err := modelrecipe.ActivateCapability(t.Context(), l.store, candidate, priorVerification, recipe.EvidenceParity, "attempt to reuse predecessor evidence for a different candidate"); err == nil {
			t.Fatal("candidate activated with predecessor evidence")
		}
		active, _, err := modelrecipe.ResolveActiveCapability(t.Context(), l.store, l.base.Model, recipe.TaskTranscription)
		if err != nil || active.Definition.ID != l.base.ID {
			t.Fatalf("failed activation displaced predecessor: %v", err)
		}
	}
	refuseActivation()
	if session, err := speechrecognition.LoadSession(t.Context(), l.store, candidate.ID, adapterAcceptanceMemory); err == nil {
		session.Close(t.Context())
		t.Fatal("unpublished checkpoint admitted execution")
	}
	l.commit(t, batch, nil)
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), l.store, "recovery/candidate", candidate); err != nil {
		t.Fatal(err)
	}
	// Once all dependencies exist, refusal must still preserve the predecessor:
	// its verification cannot authorize this distinct, fully published candidate.
	refuseActivation()
	alias := artifact.AliasBinding{Name: "test/audio-recovery/checkpoint", Target: checkpoint.ID()}
	l.commit(t, artifact.Batch{Key: alias.Name, Aliases: []artifact.AliasBinding{alias}}, nil)
	backup := filepath.Join(t.TempDir(), "backup")
	backupReport, err := l.store.Backup(t.Context(), backup)
	head, sequence = backupReport.Head, backupReport.Sequence
	if err != nil {
		t.Fatal(err)
	}
	// Only the new isolated backup is damaged. Discover the active journal via
	// the storage owner's layout classification, not a second filename authority.
	entries, err := os.ReadDir(backup)
	if err != nil {
		t.Fatal(err)
	}
	var journal string
	for _, entry := range entries {
		path := filepath.Join(backup, entry.Name())
		if !entry.IsDir() && overgodb.LayoutStorageClass(path) == overgodb.ClassCanonicalFact {
			if journal != "" {
				t.Fatal("ambiguous active journal")
			}
			journal = path
		}
	}
	if journal == "" {
		t.Fatal("backup has no active journal")
	}
	file, err := os.OpenFile(journal, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := file.Write([]byte{0}) // A strict prefix of a record header.
	if err := errors.Join(writeErr, file.Close()); err != nil {
		t.Fatal(err)
	}
	recovered, err := overgodb.Open(backup)
	if err != nil {
		t.Fatal(err)
	}
	defer recovered.Close()
	if actual, count := recovered.Head(); actual != head || count != sequence {
		t.Fatal("torn-tail recovery changed committed prefix")
	}
	assertCheckpoint := func(store *overgodb.Store) {
		id, found, err := store.ResolveAlias(t.Context(), alias.Name)
		if err != nil || !found || id != checkpoint.ID() {
			t.Fatalf("checkpoint alias not restored: %v", err)
		}
		content, err := artifact.RequireTypedContent(t.Context(), store, id)
		if err != nil {
			t.Fatal(err)
		}
		actual, err := trainingprogram.ParseCheckpoint(content.Data)
		if err != nil || actual.ID() != checkpoint.ID() || actual.Stream != checkpoint.Stream || !reflect.DeepEqual(actual.Optimizer, checkpoint.Optimizer) {
			t.Fatalf("restored checkpoint state differs: %v", err)
		}
	}
	assertCheckpoint(recovered)
	compacted := filepath.Join(t.TempDir(), "compacted")
	report, err := overgodb.Compact(t.Context(), recovered, compacted, nil)
	if err != nil || report.RetainedArtifacts == 0 {
		t.Fatalf("audio store compaction: %v", err)
	}
	retained, err := overgodb.OpenReadOnly(compacted)
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Close()
	assertCheckpoint(retained)
	loaded, err := speechrecognition.LoadSession(t.Context(), retained, candidate.ID, adapterAcceptanceMemory)
	if err != nil {
		t.Fatalf("compacted audio candidate cannot reload: %v", err)
	}
	if err := loaded.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if current, count := l.store.Head(); current != head || count != sequence {
		t.Fatal("recovery drill modified source store")
	}
	// A rejected update can resume only through the existing exact authority.
	uninterrupted := l.update(t, l.batcher(t, &trainingdata.StreamState{Identity: checkpoint.Stream.Identity, Position: checkpoint.Stream.Position}))
	expectedWeights := l.adapter.WeightSnapshot()
	expectedOptimizer, err := l.adapter.OptimizerSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	l.spec.Initial = trainingprogram.InitialStateSpec{Checkpoint: checkpoint.ID()}
	plan, err := trainingprogram.CompileTrainingRunPlanFromRepository(t.Context(), l.store, l.spec)
	if err != nil {
		t.Fatal(err)
	}
	_, err = trainingworkflow.RestoreInputProjection(directory, l.adapter, l.binding, plan,
		trainingprogram.ResumeAuthority{Model: l.base.Model, Stream: checkpoint.Stream, OptimizerPlan: l.adapter.Program().OptimizerIdentity()}, l.rng(checkpoint.Stream.Position))
	if err != nil {
		t.Fatal(err)
	}
	next := l.update(t, l.batcher(t, &trainingdata.StreamState{Identity: checkpoint.Stream.Identity, Position: checkpoint.Stream.Position}))
	actualOptimizer, err := l.adapter.OptimizerSnapshot()
	if err != nil || next != uninterrupted || !slices.Equal(expectedWeights, l.adapter.WeightSnapshot()) || !reflect.DeepEqual(expectedOptimizer, actualOptimizer) {
		t.Fatal("recovered update differs in source, loss, weights or optimizer")
	}
	t.Log("real adapter weight-write and store-publication interruption, preserved predecessor, isolated backup torn-tail recovery and compaction, exact checkpoint state and resumed source/update; canonical user stores unchanged")
}
