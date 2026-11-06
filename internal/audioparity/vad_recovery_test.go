package audioparity

import (
	"context"
	"errors"
	"io"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/recipecontract"
	"overgo/internal/speechactivity"
	"overgo/internal/testcheck"
	"overgo/internal/workflowruntime"
)

var errVADPublication = errors.New("injected VAD publication failure")

type vadFaultRepository struct {
	artifact.Repository
	fail          bool
	cancel        context.CancelCauseFunc
	target        artifact.ID
	hidden        artifact.ID
	manifestDrift bool
}

func (r *vadFaultRepository) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if r.fail {
		return artifact.CommitID{}, errVADPublication
	}
	return r.Repository.Commit(ctx, batch)
}

func (r *vadFaultRepository) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	if id == r.hidden {
		return artifact.Descriptor{}, nil, false, nil
	}
	desc, reader, found, err := r.Repository.OpenContent(ctx, id)
	if id == r.target && r.cancel != nil {
		r.cancel(context.Canceled)
	}
	return desc, reader, found, err
}

func (r *vadFaultRepository) Manifest(ctx context.Context, id artifact.ID) (artifact.Manifest, bool, error) {
	manifest, found, err := r.Repository.Manifest(ctx, id)
	if r.manifestDrift && found && len(manifest.Components) > 0 {
		manifest.Components = slices.Clone(manifest.Components)
		manifest.Components[0].Name += "-altered"
	}
	return manifest, found, err
}

// verifyVADStreamRecovery runs once, as TestVADStreamingAcceptance's
// recovery-and-discontinuity subtest.
func verifyVADStreamRecovery(t *testing.T) {
	t.Parallel()
	reference, store, audio := loadVADReference(t)
	l := newVADLifecycle(t, store, reference.Models[1], true)
	fault := &vadFaultRepository{Repository: l.store}
	detector := testcheck.Must(speechactivity.LoadDetector(t.Context(), fault, l.recipe, vadReferenceBytes))(t)
	var workspace speechactivity.DetectionWorkspace
	processor := capabilityruntime.AudioStreamProcessor(func(ctx context.Context, _ int, work workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
		return detector.ProcessStream(ctx, work, &workspace)
	})
	director := testcheck.Must(capabilityruntime.NewResidentModelSessionDirector("activity-recovery", "cpu", 1, processor))(t)
	chunk := len(audio.Samples) / len(reference.Models)
	batch, err := (workflowruntime.AudioStreamPolicy{Model: l.profile.Model, Format: audio.Format, MaxChunkSamples: uint64(len(audio.Samples))}).Batch("vad/recovery-policy")
	l.commit(t, batch, err)
	source := recipecontract.AudioReference{Audio: l.audio(t, audio.Samples, int(audio.Format.SampleRate)), Profile: batch.Contents[0].Descriptor.ID}
	first := workflowruntime.AudioStreamChunk{Audio: l.audio(t, audio.Samples[:chunk], int(audio.Format.SampleRate)), Span: recipecontract.SampleSpan{End: uint64(chunk)}}
	last := workflowruntime.AudioStreamChunk{Sequence: 1, Audio: l.audio(t, audio.Samples[chunk:], int(audio.Format.SampleRate)), Span: recipecontract.SampleSpan{Start: uint64(chunk), End: uint64(len(audio.Samples))}, Final: true}
	session := testcheck.Must(capabilityruntime.OpenAudioStream(t.Context(), l.store, residentAudioAdmission(director), source, artifact.ID{}))(t)
	defer func() { _ = session.Close(t.Context()) }()
	if _, err := session.Process(t.Context(), first); err != nil {
		t.Fatal(err)
	}
	batch, err = session.CheckpointBatch("vad/recovery-checkpoint")
	l.commit(t, batch, err)
	checkpoint := batch.Contents[0].Descriptor.ID
	testcheck.NoError(t, session.Close(t.Context()))
	for _, mode := range []string{"cancel", "publication"} {
		t.Run(mode, func(t *testing.T) {
			workspace = speechactivity.DetectionWorkspace{}
			session = testcheck.Must(capabilityruntime.OpenAudioStream(t.Context(), l.store, residentAudioAdmission(director), source, checkpoint))(t)
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			if mode == "cancel" {
				fault.target, fault.cancel = last.Audio, cancel
			} else {
				fault.fail = true
			}
			result, err := session.Process(ctx, last)
			want := errVADPublication
			if mode == "cancel" {
				want = context.Canceled
			}
			if !errors.Is(err, want) || result != (workflowruntime.AudioStreamResult{}) || director.Available() != 1 {
				t.Fatalf("failure did not close/release without output: %v", err)
			}
			fault.fail, fault.cancel = false, nil
			preserved, err := session.CheckpointBatch("vad/recovery-checkpoint")
			if err != nil || preserved.Contents[0].Descriptor.ID != checkpoint {
				t.Fatal("failed chunk advanced durable checkpoint")
			}
		})
	}
	workspace = speechactivity.DetectionWorkspace{}
	session = testcheck.Must(capabilityruntime.OpenAudioStream(t.Context(), l.store, residentAudioAdmission(director), source, checkpoint))(t)
	bad := last
	bad.Span.Start++
	if _, err := session.Process(t.Context(), bad); err == nil || director.Available() != 0 {
		t.Fatal("gap admission changed live lease")
	}
	// A declared discontinuity begins a new analysis segment at this offset.
	last.Discontinuity = true
	result := testcheck.Must(session.Process(t.Context(), last))(t)
	output := testcheck.Must(requireActivity(t.Context(), l.store, result.Output))(t)
	for _, segment := range output.Segments {
		if segment.Span.Start < uint64(chunk) || segment.Span.End > uint64(len(audio.Samples)) {
			t.Fatal("discontinuity retained prior segment offsets")
		}
	}
	// Compare the same segment with a fresh cursor starting at the discontinuity.
	workspace = speechactivity.DetectionWorkspace{}
	fresh := testcheck.Must(capabilityruntime.OpenAudioStream(t.Context(), l.store, residentAudioAdmission(director), source, artifact.ID{}))(t)
	defer fresh.Close(t.Context())
	last.Sequence = 0
	replayed := testcheck.Must(fresh.Process(t.Context(), last))(t)
	freshOutput, err := requireActivity(t.Context(), l.store, replayed.Output)
	if err != nil || !reflect.DeepEqual(output.Segments, freshOutput.Segments) {
		t.Fatal("discontinuity differs from fresh analysis")
	}
	t.Log("real CPU model: cancellation during source read, publication failure, unchanged checkpoint, lease release, gap refusal and discontinuity/fresh equivalence executed")
}
