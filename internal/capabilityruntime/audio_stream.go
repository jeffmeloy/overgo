package capabilityruntime

import (
	"context"
	"errors"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/workflowruntime"
)

// ErrAudioBackpressure means a chunk is already executing; no input is queued.
var ErrAudioBackpressure = errors.New("audio stream: chunk already executing")

// AudioStreamProcessor executes synchronously on one leased slot. It must honor
// cancellation, restore Work.PreviousState (or reset), and publish complete
// output/state artifacts before returning. It must not retain input buffers or
// use a slot after returning. Model-specific numerical work is outside this owner.
type AudioStreamProcessor func(context.Context, int, workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error)

// AudioStreamLease binds synchronous processing to an already admitted model
// slot. Release must be idempotent. The admitting owner retains its existing
// residency policy; a stream does not create another director or worker.
type AudioStreamLease struct {
	Processor AudioStreamProcessor
	Slot      int
	Release   func() error
}

// AudioStreamSession admits one chunk at a time without a worker, queue or
// accumulated waveform. Close joins the active call before releasing residency;
// a timed-out Close leaves release owned by that call's cleanup. Do not copy it.
type AudioStreamSession struct {
	mu     sync.Mutex
	cursor *workflowruntime.AudioStreamCursor
	reader artifact.Reader
	lease  AudioStreamLease
	cancel context.CancelCauseFunc
	done   chan struct{}
	closed bool
}

// OpenAudioStream validates restart authority and leases existing bounded model
// residency. The caller must Close even if it never submits a chunk.
func OpenAudioStream(ctx context.Context, reader artifact.Reader, acquire func(context.Context) (AudioStreamLease, error), source recipecontract.AudioReference, resume artifact.ID) (*AudioStreamSession, error) {
	if ctx == nil || reader == nil || acquire == nil {
		return nil, ErrSessionUnavailable
	}
	cursor, err := workflowruntime.LoadAudioStream(ctx, reader, source, resume)
	if err != nil {
		return nil, err
	}
	lease, err := acquire(ctx)
	if err != nil {
		return nil, err
	}
	if lease.Processor == nil || lease.Release == nil {
		if lease.Release != nil {
			return nil, errors.Join(ErrSessionUnavailable, lease.Release())
		}
		return nil, ErrSessionUnavailable
	}
	return &AudioStreamSession{cursor: cursor, reader: reader, lease: lease}, nil
}

// Process invokes one bounded chunk on the caller goroutine. Invalid admission
// leaves progress intact; execution failure/cancellation closes this session and
// preserves only its last completed checkpoint. A final chunk closes on success.
func (s *AudioStreamSession) Process(ctx context.Context, chunk workflowruntime.AudioStreamChunk) (result workflowruntime.AudioStreamResult, err error) {
	if s == nil || ctx == nil {
		return result, ErrSessionUnavailable
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	s.mu.Lock()
	if s.closed || s.lease.Processor == nil {
		s.mu.Unlock()
		return result, ErrSessionUnavailable
	}
	if s.done != nil {
		s.mu.Unlock()
		return result, ErrAudioBackpressure
	}
	work, err := s.cursor.Prepare(chunk)
	if err != nil {
		s.mu.Unlock()
		return result, err
	}
	callCtx, cancel := context.WithCancelCause(ctx)
	done := make(chan struct{})
	s.cancel, s.done = cancel, done
	s.mu.Unlock()
	committed := false
	defer func() {
		cancel(err)
		s.mu.Lock()
		s.closed = s.closed || !committed
		if s.closed {
			s.mu.Unlock()
			err = errors.Join(err, s.lease.Release())
			s.mu.Lock()
		}
		s.cancel, s.done = nil, nil
		close(done)
		s.mu.Unlock()
	}()
	if chunk.Audio.Valid() {
		if _, found, err := s.reader.Artifact(callCtx, chunk.Audio); err != nil || !found {
			return result, errors.Join(errors.New("audio stream: chunk artifact is absent"), err)
		}
	}
	result, err = s.lease.Processor(callCtx, s.lease.Slot, work)
	if err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	if err := result.Validate(); err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	for _, id := range []artifact.ID{result.Output, result.State} {
		if _, found, err := s.reader.Artifact(callCtx, id); err != nil || !found {
			return workflowruntime.AudioStreamResult{}, errors.Join(errors.New("audio stream: unpublished result"), err)
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := callCtx.Err(); err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	if s.closed {
		return workflowruntime.AudioStreamResult{}, ErrSessionUnavailable
	}
	if err := s.cursor.Complete(work, result); err != nil {
		return workflowruntime.AudioStreamResult{}, err
	}
	committed, s.closed = true, chunk.Final
	return result, nil
}

// CheckpointBatch snapshots only a completed boundary, including after failure
// or finalization. The caller publishes it through the artifact repository.
func (s *AudioStreamSession) CheckpointBatch(key string) (artifact.Batch, error) {
	if s == nil {
		return artifact.Batch{}, ErrSessionUnavailable
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done != nil {
		return artifact.Batch{}, ErrAudioBackpressure
	}
	return s.cursor.Batch(key)
}

// Close cancels and joins active execution. It is idempotent; a cancelled close
// context does not release a slot while its processor is still executing.
func (s *AudioStreamSession) Close(ctx context.Context) error {
	if s == nil || ctx == nil {
		return ErrSessionUnavailable
	}
	s.mu.Lock()
	s.closed = true
	cancel, done, lease := s.cancel, s.done, s.lease
	s.mu.Unlock()
	if cancel != nil {
		cancel(ErrSessionUnavailable)
	}
	if done != nil {
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if lease.Release == nil {
		return ErrSessionUnavailable
	}
	return lease.Release()
}
