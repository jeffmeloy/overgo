package capabilityruntime_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/testevidence"
	"overgo/internal/workflowruntime"
)

func TestNativeASRSessionAcceptance(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": native recurrent sessions require registered model and corpus captures")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	fixture := speechrecognitiontest.PublishNative(t, store)
	for _, clip := range fixture.Clips {
		t.Run(clip.Name, func(t *testing.T) { verifyNativeSession(t, store, fixture, clip) })
	}
	t.Run("reset-overlap-empty-final", func(t *testing.T) {
		verifyNativeStreamBoundaries(t, store, fixture, fixture.Clips[0])
	})
}

func verifyNativeStreamBoundaries(t *testing.T, store *overgodb.Store, fixture speechrecognitiontest.NativeFixture, clip speechrecognitiontest.NativeClip) {
	t.Helper()
	rate := fixture.Profile.Frontend.SampleRate
	hop := int(fixture.Profile.Frontend.Geometry.HopSamples)
	_, chunks := fixture.PublishStream(t, store, clip, rate+1)
	withPrefix := clip
	withPrefix.PCM = append(slices.Clone(clip.PCM[:chunks[0].Span.End]), clip.PCM...)
	source, _ := fixture.PublishStream(t, store, withPrefix, rate+1)
	policy := workflowruntime.AudioStreamPolicy{Model: fixture.Definition.Model,
		Format:          recipecontract.AudioFormat{SampleRate: uint64(rate), Channels: 1, Encoding: "pcm-f32le"},
		MaxChunkSamples: uint64(rate + 1 + hop), MaxOverlapSamples: uint64(hop)}
	batch, err := policy.Batch("native-stream/overlap-policy")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		t.Fatal(err)
	}
	source.Profile = batch.Contents[0].Descriptor.ID
	session, err := speechrecognition.LoadSession(t.Context(), store, fixture.Definition.ID, 4<<30)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { session.Close(context.WithoutCancel(t.Context())) })
	stream, err := session.OpenStream(t.Context(), speechrecognition.StreamRequest{Source: source})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { stream.Close(context.WithoutCancel(t.Context())) })
	// Execute a prefix, then explicitly abandon its acoustic and text state.
	if _, err := stream.Process(t.Context(), chunks[0]); err != nil {
		t.Fatal(err)
	}
	origin := chunks[0].Span.End
	var text strings.Builder
	for i, original := range chunks {
		chunk := original
		chunk.Sequence++
		chunk.Discontinuity, chunk.Final = i == 0, false
		if i != 0 {
			chunk.Span.Start -= uint64(hop)
			data := speechrecognitiontest.NativeWave(t, clip.PCM[chunk.Span.Start:chunk.Span.End], rate)
			id, err := artifact.IdentifyBytes(artifact.KindFile, data)
			if err != nil {
				t.Fatal(err)
			}
			payload := artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data))}, Data: data}
			if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "native-stream/overlap/" + id.String(), Contents: []artifact.Content{payload}}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
				t.Fatal(err)
			}
			chunk.Audio = id
		}
		chunk.Span.Start += origin
		chunk.Span.End += origin
		result, err := stream.Process(t.Context(), chunk)
		if err != nil {
			t.Fatal(err)
		}
		output, err := speechrecognition.RequireTranscriptionChunk(t.Context(), store, result.Output)
		if err != nil || output.Span.Start != original.Span.Start+origin {
			t.Fatalf("overlap was not excluded: %+v, %v", output.Span, err)
		}
		text.WriteString(output.Text)
	}
	end := origin + uint64(len(clip.PCM))
	result, err := stream.Process(t.Context(), workflowruntime.AudioStreamChunk{Sequence: uint64(len(chunks) + 1), Span: recipecontract.SampleSpan{Start: end, End: end}, Final: true})
	if err != nil {
		t.Fatal(err)
	}
	output, err := speechrecognition.RequireTranscriptionChunk(t.Context(), store, result.Output)
	if err != nil || !output.Final || output.Span.Start != end || output.Span.End != end {
		t.Fatalf("empty finalization differs: %+v, %v", output, err)
	}
	text.WriteString(output.Text)
	if text.String() != clip.Streaming || session.Snapshot().Active != 0 {
		t.Fatalf("reset/overlap/final transcript=%q want=%q or lease retained", text.String(), clip.Streaming)
	}
}

func verifyNativeSession(t *testing.T, store *overgodb.Store, fixture speechrecognitiontest.NativeFixture, clip speechrecognitiontest.NativeClip) {
	t.Helper()
	commit := func(batch artifact.Batch, err error) {
		t.Helper()
		if err == nil {
			_, err = artifact.CommitBatch(t.Context(), store, batch)
		}
		if err != nil && !errors.Is(err, artifact.ErrNoChange) {
			t.Fatal(err)
		}
	}
	source, chunks := fixture.PublishStream(t, store, clip, fixture.Profile.Frontend.SampleRate+1)
	var reference []workflowruntime.AudioStreamResult
	for _, restart := range []bool{false, true} {
		session, err := speechrecognition.LoadSession(t.Context(), store, fixture.Definition.ID, 4<<30)
		if err != nil {
			t.Fatal(err)
		}
		stream, err := session.OpenStream(t.Context(), speechrecognition.StreamRequest{Source: source, Resume: artifact.ID{}})
		if err != nil {
			t.Fatal(err)
		}
		var text strings.Builder
		var results []workflowruntime.AudioStreamResult
		started := time.Now()
		var firstPartial time.Duration
		for index, chunk := range chunks {
			if index == 1 && restart {
				batch, err := stream.CheckpointBatch("native-stream/restart/" + clip.Name)
				commit(batch, err)
				if err := stream.Close(t.Context()); err != nil {
					t.Fatal(err)
				}
				if err := session.Close(t.Context()); err != nil {
					t.Fatal(err)
				}
				session, err = speechrecognition.LoadSession(t.Context(), store, fixture.Definition.ID, 4<<30)
				if err != nil {
					t.Fatal(err)
				}
				stream, err = session.OpenStream(t.Context(), speechrecognition.StreamRequest{Source: source, Resume: batch.Contents[0].Descriptor.ID})
				if err != nil {
					t.Fatal(err)
				}
			}
			bad := chunk
			bad.Sequence++
			if _, err := stream.Process(t.Context(), bad); err == nil {
				t.Fatal("out-of-order input was accepted")
			}
			result, err := stream.Process(t.Context(), chunk)
			if err != nil {
				t.Fatal(err)
			}
			output, err := speechrecognition.RequireTranscriptionChunk(t.Context(), store, result.Output)
			if err != nil || output.Source != source || output.Recipe != fixture.Definition.ID || output.Sequence != chunk.Sequence || output.State != result.State || output.Final != chunk.Final {
				t.Fatalf("stream output authority differs: %v", err)
			}
			text.WriteString(output.Text)
			if output.Text != "" && firstPartial == 0 {
				firstPartial = time.Since(started)
			}
			results = append(results, result)
		}
		if text.String() != clip.Streaming || firstPartial == 0 {
			t.Fatalf("native stream transcript=%q want=%q", text.String(), clip.Streaming)
		}
		if restart && !slices.Equal(results, reference) {
			t.Fatal("checkpoint restart changed output or numerical-state artifact identity")
		}
		reference = results
		if session.Snapshot().Active != 0 {
			t.Fatal("finalization retained an active component lease")
		}
		if err := stream.Close(t.Context()); err != nil {
			t.Fatal(err)
		}
		if err := session.Close(context.WithoutCancel(t.Context())); err != nil {
			t.Fatal(err)
		}
		if snapshot := session.Snapshot(); !snapshot.Closed || snapshot.Active != 0 || snapshot.Entries != 0 {
			t.Fatalf("session did not drain: %+v", snapshot)
		}
		t.Logf("restart=%v waveform chunks=%d first-text=%s completed=%s source-seconds=%.3f; includes store publication and reload, not isolated inference timing; CPU only", restart, len(chunks), firstPartial, time.Since(started), float64(len(clip.PCM))/float64(fixture.Profile.Frontend.SampleRate))
	}
}
