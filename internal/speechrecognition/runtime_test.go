package speechrecognition

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/recipecontract"
)

func BenchmarkRecognizerWaveform(b *testing.B) {
	directory, declaration := transcriptionFixtureModel(b)
	frontend := audiodsp.FrontendConfig{
		SampleRate: 16_000, FFTLength: 512, FrameSpan: 400,
		Geometry: recipecontract.AudioFrameGeometry{WindowSamples: 400, HopSamples: 160, FeatureBins: 4},
		Padding:  "zero", Window: "periodic-hann",
		Mel: audiodsp.MelConfig{Scale: "htk", MinFrequency: 20, MaxFrequency: 7_000},
		Log: audiodsp.LogConfig{Power: true, Base: "natural", GuardMode: "clamp", Guard: 1e-8, Scale: 1},
	}
	model, err := LoadRecognizer(b.Context(), directory, RecognizerSpec{
		Declaration: declaration, Frontend: frontend,
		Grouping: audiodsp.GroupedFeatureConfig{StackFrames: 1, FinalFrameSamples: 400},
	}, 1<<24)
	if err != nil {
		b.Fatal(err)
	}
	request := TranscriptionRequest{Audio: transcriptionWAV(testWave(frontend.SampleRate)), MaximumSamples: uint64(frontend.SampleRate), MemoryBytes: 1 << 24}
	want, err := model.Transcribe(b.Context(), request)
	if err != nil || want == "" {
		b.Fatalf("fixture transcript %q: %v", want, err)
	}
	b.Logf("synthetic waveform transcript=%q", want)
	b.ReportAllocs()
	for b.Loop() {
		if got, err := model.Transcribe(b.Context(), request); err != nil || got != want {
			b.Fatalf("transcript %q, want %q: %v", got, want, err)
		}
	}
}

func TestRecognizerLoadAdmission(t *testing.T) {
	t.Parallel()
	canceled, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	directory := filepath.Join(t.TempDir(), "absent-checkpoint")
	for _, form := range []struct {
		name    string
		binding *TransducerBinding
	}{{"connectionist", nil}, {"recurrent", &TransducerBinding{}}} {
		t.Run(form.name, func(t *testing.T) {
			for _, test := range []struct {
				name    string
				ctx     context.Context
				memory  uint64
				message string
			}{
				{"canceled", canceled, 1, ""},
				{"nil-context", nil, 1, "context and host memory ceiling"},
				{"zero-memory", t.Context(), 0, "context and host memory ceiling"},
			} {
				t.Run(test.name, func(t *testing.T) {
					model, err := LoadRecognizer(test.ctx, directory, RecognizerSpec{Transducer: form.binding}, test.memory)
					if model != nil || err == nil {
						t.Fatalf("inadmissible load returned model %v and error %v", model, err)
					}
					if test.message == "" {
						if !errors.Is(err, context.Canceled) {
							t.Fatalf("canceled load accessed checkpoint: %v", err)
						}
					} else if !strings.Contains(err.Error(), test.message) {
						t.Fatalf("invalid admission accessed checkpoint: %v", err)
					}
				})
			}
		})
	}
}

// TestTranscriptStageEncodesItsOutput holds the transcription stage to the
// rule every scalar capability stage answers to: what it returns leaves the
// runtime with a content identity. A stage registered without an encoder
// executes a model to completion and is then refused at the last step, so the
// encoding is checked here rather than discovered by a verify run.
func TestTranscriptStageEncodesItsOutput(t *testing.T) {
	t.Parallel()
	for _, text := range []string{"a transcript", ""} {
		content, err := transcriptContent(text)
		if err != nil {
			t.Fatal(err)
		}
		if err := content.Validate(); err != nil {
			t.Fatal(err)
		}
		if content.Descriptor.ID.Kind() != artifact.KindFile {
			t.Fatalf("transcript identity kind = %q", content.Descriptor.ID.Kind())
		}
	}
	first, err := transcriptContent("one")
	if err != nil {
		t.Fatal(err)
	}
	second, err := transcriptContent("another")
	if err != nil {
		t.Fatal(err)
	}
	if first.Descriptor.ID == second.Descriptor.ID {
		t.Fatal("two transcripts share one identity")
	}
	if err := RegisterRuntime(nil, first.Descriptor.ID, nil); err == nil {
		t.Fatal("a runtime binding without a recognizer was accepted")
	}
}
