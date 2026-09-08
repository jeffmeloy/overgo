package audioparity

import (
	"context"
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/capabilityruntime"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/speechactivity"
	"overgo/internal/workflowruntime"
)

type vadLifecycle struct {
	audioPublication
	detector *speechactivity.Detector
	profile  speechactivity.Profile
	recipe   artifact.ID
}

func newVADLifecycle(t *testing.T, reference *overgodb.Store, model vadNumericalModel, streaming bool) *vadLifecycle {
	t.Helper()
	inventory, license, _, declaration := loadVADArtifacts(t, reference, model)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	l := &vadLifecycle{audioPublication: audioPublication{store: store}}
	batch, err := inventory.Batch("vad/model")
	l.commit(t, batch, err)
	l.commit(t, artifact.Batch{Key: "vad/license", Contents: []artifact.Content{license}, Lineage: artifact.DependencyLineage(inventory.Manifest.ID, license.Descriptor.ID)}, nil)
	var frontend audiodsp.FrontendConfig
	readVADJSON(t, "recipes/vad_frontend.json", &frontend)
	profile := speechactivity.Profile{Model: inventory.Manifest.ID, Inventory: inventory.TensorInventory.ID, License: license.Descriptor.ID, Frontend: frontend, Network: declaration}
	// Use the pinned policy fixture, including its explicit non-default gap and
	// extension settings. Policy data is not duplicated as executable defaults.
	cases := vadBoundaryCases(t)
	c := cases[len(cases)-1]
	if streaming {
		profile.Streaming = &c.Config
	} else {
		profile.Offline = &speechactivity.OfflineConfig{Smoothing: c.Config.Smoothing, Threshold: c.Config.Threshold, MinSpeech: int(c.Config.MinSpeech), MaxSpeech: int(c.Config.MaxSpeech), MinSilence: int(c.Config.MinSilence), MergeSilence: c.MergeSilence, ExtendSpeech: c.ExtendSpeech}
	}
	l.profile, err = speechactivity.NewProfile(profile)
	if err != nil {
		t.Fatal(err)
	}
	batch, err = l.profile.Batch("vad/profile")
	l.commit(t, batch, err)
	definition, err := modelrecipe.ActivityDefinition(profile.Model, l.profile.ID, profile.Inventory)
	if err != nil {
		t.Fatal(err)
	}
	content, err := definition.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	l.commit(t, artifact.Batch{Key: "vad/recipe", Contents: []artifact.Content{content}, Lineage: artifact.DependencyLineage(definition.ID, profile.Model, l.profile.ID, profile.Inventory)}, nil)
	l.recipe = definition.ID
	l.detector, err = speechactivity.LoadDetector(t.Context(), store, definition.ID, vadReferenceBytes)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func (l *vadLifecycle) audio(t *testing.T, samples []float32, rate int) artifact.ID {
	t.Helper()
	encoded, err := media.EncodeWAVPCM16(samples, rate)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := media.DecodeAudio(t.Context(), encoded, uint64(len(samples)))
	if err != nil || !slices.Equal(samples, decoded.Samples) {
		t.Fatal("test PCM materialization changed corpus samples")
	}
	id, err := artifact.IdentifyBytes(artifact.KindFile, encoded)
	if err != nil {
		t.Fatal(err)
	}
	l.commit(t, artifact.Batch{Key: id.String(), Contents: []artifact.Content{{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(encoded)), MediaType: media.WAVMediaType}, Data: encoded}}}, nil)
	return id
}

func TestVADOfflineAcceptance(t *testing.T) {
	t.Run("artifact-refusals", TestVADArtifactRefusals)
	t.Run("numerical-reference", TestVADNumericalParity)
	t.Run("boundary-reference", TestVADOfflineBoundaryParity)
	reference, store, audio := loadVADReference(t)
	l := newVADLifecycle(t, store, reference.Models[0], false)
	for _, name := range []string{"speech", "silence"} {
		t.Run(name, func(t *testing.T) {
			wave := audio.Samples
			if name == "silence" {
				wave = make([]float32, len(wave))
			}
			source := recipecontract.AudioReference{Audio: l.audio(t, wave, int(audio.Format.SampleRate)), Profile: l.profile.ID}
			var workspace speechactivity.DetectionWorkspace
			result, id, err := l.detector.Detect(t.Context(), source, &workspace)
			if err != nil {
				t.Fatal(err)
			}
			stored, err := speechactivity.RequireActivity(t.Context(), l.store, id)
			if err != nil || stored.Source != source || !slices.Equal(stored.Segments, result.Segments) {
				t.Fatal("offline publication differs")
			}
			if name == "speech" && len(result.Segments) == 0 || name == "silence" && len(result.Segments) != 0 {
				t.Fatal("speech/silence control differs")
			}
			ctx, cancel := context.WithCancelCause(t.Context())
			cancel(context.Canceled)
			if _, id, err := l.detector.Detect(ctx, source, &workspace); !errors.Is(err, context.Canceled) || id.Valid() {
				t.Fatal("canceled detection published a result")
			}
			t.Logf("CPU samples=%d published segments=%d; no annotated-quality or GPU claim", len(wave), len(result.Segments))
		})
	}
}

func TestVADStreamingAcceptance(t *testing.T) {
	t.Run("recovery-and-discontinuity", TestVADStreamRecovery)
	t.Run("waveform-and-cache-reference", TestVADWaveformStreamParity)
	t.Run("boundary-reference", TestVADStreamBoundaryParity)
	reference, store, audio := loadVADReference(t)
	l := newVADLifecycle(t, store, reference.Models[1], true)
	for _, name := range []string{"speech", "silence"} {
		t.Run(name, func(t *testing.T) {
			wave := audio.Samples
			if name == "silence" {
				wave = make([]float32, len(wave))
			}
			source := recipecontract.AudioReference{Audio: l.audio(t, wave, int(audio.Format.SampleRate))}
			chunk := len(wave) / len(reference.Models)
			policy := workflowruntime.AudioStreamPolicy{Model: l.profile.Model, Format: audio.Format, MaxChunkSamples: uint64(chunk + int(l.profile.Frontend.Geometry.HopSamples)), MaxOverlapSamples: l.profile.Frontend.Geometry.HopSamples}
			batch, err := policy.Batch("vad/stream-policy")
			l.commit(t, batch, err)
			source.Profile = batch.Contents[0].Descriptor.ID
			var workspace speechactivity.DetectionWorkspace
			processor := capabilityruntime.AudioStreamProcessor(func(ctx context.Context, _ int, work workflowruntime.AudioStreamWork) (workflowruntime.AudioStreamResult, error) {
				return l.detector.ProcessStream(ctx, work, &workspace)
			})
			director, err := capabilityruntime.NewResidentModelSessionDirector("activity", "cpu", 1, processor)
			if err != nil {
				t.Fatal(err)
			}
			session, err := capabilityruntime.OpenAudioStream(t.Context(), l.store, director, source, artifact.ID{})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = session.Close(t.Context()) }()
			first := workflowruntime.AudioStreamChunk{Audio: l.audio(t, wave[:chunk], int(audio.Format.SampleRate)), Span: recipecontract.SampleSpan{End: uint64(chunk)}}
			if _, err := session.Process(t.Context(), first); err != nil {
				t.Fatal(err)
			}
			batch, err = session.CheckpointBatch("vad/stream-restart/" + source.Audio.String())
			l.commit(t, batch, err)
			if err := session.Close(t.Context()); err != nil {
				t.Fatal(err)
			}
			if director.Available() != 1 {
				t.Fatal("residency leaked at checkpoint close")
			}
			workspace = speechactivity.DetectionWorkspace{}
			session, err = capabilityruntime.OpenAudioStream(t.Context(), l.store, director, source, batch.Contents[0].Descriptor.ID)
			if err != nil {
				t.Fatal(err)
			}
			start := chunk - int(policy.MaxOverlapSamples)
			last := workflowruntime.AudioStreamChunk{Sequence: 1, Audio: l.audio(t, wave[start:], int(audio.Format.SampleRate)), Span: recipecontract.SampleSpan{Start: uint64(start), End: uint64(len(wave))}, Final: true}
			result, err := session.Process(t.Context(), last)
			if err != nil {
				t.Fatal(err)
			}
			output, err := speechactivity.RequireActivity(t.Context(), l.store, result.Output)
			if err != nil || output.State != result.State {
				t.Fatal("stream publication differs")
			}
			if name == "speech" && len(output.Segments) == 0 || name == "silence" && len(output.Segments) != 0 {
				t.Fatal("stream speech/silence control differs")
			}
			if director.Available() != 1 {
				t.Fatal("finalization leaked residency")
			}
			if _, err := session.Process(t.Context(), last); !errors.Is(err, capabilityruntime.ErrSessionUnavailable) {
				t.Fatal("final session accepted more input")
			}
			t.Logf("CPU samples=%d chunks=2 checkpoint/restart=1 overlap=%d final segments=%d; real model through OpenAudioStream", len(wave), policy.MaxOverlapSamples, len(output.Segments))
		})
	}
}
