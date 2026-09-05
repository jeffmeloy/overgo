package capabilityruntime

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

type audioSessionFixture struct {
	store  *overgodb.Store
	source recipecontract.AudioReference
	result workflowruntime.AudioStreamResult
	chunk  workflowruntime.AudioStreamChunk
}

func newAudioSessionFixture(t *testing.T) audioSessionFixture {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	model := testutil.ArtifactID(t, artifact.KindModel, "leased-model")
	f := audioSessionFixture{store: store}
	f.source.Audio = testutil.ArtifactID(t, artifact.KindFile, "audio")
	f.result = workflowruntime.AudioStreamResult{
		Output: testutil.ArtifactID(t, artifact.KindOutput, "transcript"),
		State:  testutil.ArtifactID(t, artifact.KindCheckpoint, "decoder-state"),
	}
	for _, id := range []artifact.ID{model, f.source.Audio, f.result.Output, f.result.State} {
		testutil.PublishArtifact(t, store, id)
	}
	batch, err := (workflowruntime.AudioStreamPolicy{
		Model: model, Format: recipecontract.AudioFormat{SampleRate: 16_000, Channels: 1, Encoding: "pcm-f32le"},
		MaxChunkSamples: 8, MaxOverlapSamples: 2,
	}).Batch("policy")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	f.source.Profile = batch.Contents[0].Descriptor.ID
	f.chunk = workflowruntime.AudioStreamChunk{Audio: f.source.Audio, Span: recipecontract.SampleSpan{End: 8}}
	return f
}

func (f audioSessionFixture) open(t *testing.T, processor AudioStreamProcessor) (*AudioStreamSession, *ModelSessionDirector[struct{}, AudioStreamProcessor, struct{}]) {
	t.Helper()
	director, err := NewResidentModelSessionDirector("audio", "cpu", 1, processor)
	if err != nil {
		t.Fatal(err)
	}
	session, err := OpenAudioStream(t.Context(), f.store, director, f.source, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { session.Close(t.Context()) })
	return session, director
}

func TestAudioSessionRestartAndFinalization(t *testing.T) {
	f := newAudioSessionFixture(t)
	var calls []workflowruntime.AudioStreamWork
	session, director := f.open(t, func(_ context.Context, slot int, work workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
		if slot != 0 {
			t.Errorf("slot=%d", slot)
		}
		calls = append(calls, work)
		return f.result, nil
	})
	if result, err := session.Process(t.Context(), f.chunk); err != nil || result != f.result {
		t.Fatalf("first result: %+v %v", result, err)
	}
	batch, err := session.CheckpointBatch("first")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), f.store, batch); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	restored, err := OpenAudioStream(t.Context(), f.store, director, f.source, batch.Contents[0].Descriptor.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close(t.Context())
	final := workflowruntime.AudioStreamChunk{Sequence: 1, Span: recipecontract.SampleSpan{Start: 6, End: 14}, Audio: f.source.Audio, Final: true}
	if _, err := restored.Process(t.Context(), final); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || !calls[0].Reset || calls[1].Reset || calls[1].PreviousState != f.result.State ||
		calls[1].NewSpan != (recipecontract.SampleSpan{Start: 8, End: 14}) || director.Available() != 1 {
		t.Fatalf("restart/final: calls=%+v available=%d", calls, director.Available())
	}
	if _, err := restored.Process(t.Context(), final); !errors.Is(err, ErrSessionUnavailable) {
		t.Fatalf("final stream admitted input: %v", err)
	}
	batch, err = restored.CheckpointBatch("final")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), f.store, batch); err != nil {
		t.Fatal(err)
	}
	cursor, err := workflowruntime.LoadAudioStream(t.Context(), f.store, f.source, batch.Contents[0].Descriptor.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cursor.Prepare(workflowruntime.AudioStreamChunk{Sequence: 2, Final: true}); err == nil {
		t.Fatal("restart lost finalization")
	}
}

func TestAudioSessionBackpressureCancellationAndLeaseLifetime(t *testing.T) {
	f := newAudioSessionFixture(t)
	started, cancelled, release := make(chan struct{}), make(chan struct{}), make(chan struct{})
	session, director := f.open(t, func(ctx context.Context, _ int, _ workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
		close(started)
		<-ctx.Done()
		close(cancelled)
		<-release
		// Even a misbehaving callback returning success after cancellation must
		// not commit progress or make its lease available early.
		return f.result, nil
	})
	before, err := session.CheckpointBatch("before")
	if err != nil {
		t.Fatal(err)
	}
	processed := make(chan error, 1)
	go func() { _, err := session.Process(t.Context(), f.chunk); processed <- err }()
	<-started
	_, pressureErr := session.Process(t.Context(), f.chunk)
	_, snapshotErr := session.CheckpointBatch("active")
	closeCtx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	closeErr := session.Close(closeCtx)
	<-cancelled
	availableWhileActive := director.Available()
	close(release)
	processErr := <-processed
	if !errors.Is(pressureErr, ErrAudioBackpressure) || !errors.Is(snapshotErr, ErrAudioBackpressure) ||
		!errors.Is(closeErr, context.Canceled) || !errors.Is(processErr, context.Canceled) || availableWhileActive != 0 {
		t.Fatalf("pressure=%v snapshot=%v close=%v process=%v active-slots=%d", pressureErr, snapshotErr, closeErr, processErr, availableWhileActive)
	}
	if err := session.Close(t.Context()); err != nil || director.Available() != 1 {
		t.Fatalf("close did not release lease: %v available=%d", err, director.Available())
	}
	after, err := session.CheckpointBatch("after")
	if err != nil || before.Contents[0].Descriptor.ID != after.Contents[0].Descriptor.ID {
		t.Fatalf("cancelled chunk advanced checkpoint: %v", err)
	}
}

func TestAudioSessionFailurePreservesCheckpoint(t *testing.T) {
	for _, name := range []string{"processor", "unpublished", "invalid-result", "missing-chunk", "caller-cancel"} {
		t.Run(name, func(t *testing.T) {
			f := newAudioSessionFixture(t)
			failure := errors.New("processor failed")
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			session, director := f.open(t, func(context.Context, int, workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
				switch name {
				case "processor":
					return workflowruntime.AudioStreamResult{}, failure
				case "invalid-result":
					return workflowruntime.AudioStreamResult{}, nil
				case "unpublished":
					return workflowruntime.AudioStreamResult{Output: testutil.ArtifactID(t, artifact.KindOutput, "absent"), State: f.result.State}, nil
				case "caller-cancel":
					cancel(nil)
				case "missing-chunk":
					t.Error("missing input invoked processor")
				}
				return f.result, nil
			})
			before, err := session.CheckpointBatch("before")
			if err != nil {
				t.Fatal(err)
			}
			if name == "missing-chunk" {
				f.chunk.Audio = testutil.ArtifactID(t, artifact.KindFile, "absent")
			}
			if _, err := session.Process(ctx, f.chunk); err == nil {
				t.Fatal("failed call accepted")
			}
			after, err := session.CheckpointBatch("after")
			if err != nil || before.Contents[0].Descriptor.ID != after.Contents[0].Descriptor.ID || director.Available() != 1 {
				t.Fatalf("failure changed progress or retained lease: %v available=%d", err, director.Available())
			}
		})
	}
}

func TestAudioSessionAdmissionAndZeroValue(t *testing.T) {
	f := newAudioSessionFixture(t)
	called := false
	session, director := f.open(t, func(context.Context, int, workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
		called = true
		return f.result, nil
	})
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := session.Process(ctx, f.chunk); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancel: %v", err)
	}
	invalid := f.chunk
	invalid.Sequence++
	if _, err := session.Process(t.Context(), invalid); err == nil {
		t.Fatal("out-of-order admitted")
	}
	if called || director.Available() != 0 {
		t.Fatal("admission executed or released session")
	}
	if _, err := session.Process(t.Context(), f.chunk); err != nil {
		t.Fatal(err)
	}
	var zero AudioStreamSession
	if _, err := zero.Process(t.Context(), f.chunk); err == nil {
		t.Fatal("zero session processed")
	}
	if _, err := zero.CheckpointBatch("zero"); err == nil {
		t.Fatal("zero session checkpointed")
	}
	if err := zero.Close(t.Context()); err == nil {
		t.Fatal("zero session closed successfully")
	}
}

func TestAudioSessionPanicReleasesLease(t *testing.T) {
	f := newAudioSessionFixture(t)
	session, director := f.open(t, func(context.Context, int, workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
		panic("processor panic")
	})
	func() {
		defer func() {
			if recover() == nil {
				t.Error("processor panic was suppressed")
			}
		}()
		_, _ = session.Process(t.Context(), f.chunk)
	}()
	if director.Available() != 1 {
		t.Fatal("panic retained lease")
	}
	if _, err := session.CheckpointBatch("after-panic"); err != nil {
		t.Fatal(err)
	}
}

func TestAudioSessionConcurrentCloseAndCompletion(t *testing.T) {
	f := newAudioSessionFixture(t)
	for range 32 {
		started, release := make(chan struct{}), make(chan struct{})
		session, director := f.open(t, func(context.Context, int, workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
			close(started)
			<-release
			return f.result, nil
		})
		processed, closed := make(chan error, 1), make(chan error, 1)
		go func() { _, err := session.Process(t.Context(), f.chunk); processed <- err }()
		<-started
		go func() { closed <- session.Close(t.Context()) }()
		close(release)
		processErr, closeErr := <-processed, <-closed
		if closeErr != nil || (processErr != nil && !errors.Is(processErr, context.Canceled) && !errors.Is(processErr, ErrSessionUnavailable)) {
			t.Fatalf("completion=%v close=%v", processErr, closeErr)
		}
		if director.Available() != 1 {
			t.Fatal("concurrent close retained lease")
		}
	}
}
