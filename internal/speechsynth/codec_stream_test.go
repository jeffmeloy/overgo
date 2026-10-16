package speechsynth

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"overgo/internal/jsonfile"
	"overgo/internal/tensor"
)

func TestMimiStreamingDecode(t *testing.T) {
	t.Run("generation boundaries", testStreamingFrameBoundaries)
	var fixture struct {
		Config    artifactConfig          `json:"config"`
		Tensors   map[string]goldenTensor `json:"tensors"`
		Mean, Std []float32
		Latents   goldenTensor
		PCM       []float64
	}
	if err := jsonfile.Decode(fixturePath(t, "streaming_codec.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	shapes := make(map[string][]int)
	for name, value := range fixture.Tensors {
		shapes[name] = value.Shape
	}
	read := func(name string, want ...int) ([]float32, error) {
		value, ok := fixture.Tensors[name]
		if !ok || !slices.Equal(value.Shape, want) {
			return nil, fmt.Errorf("fixture %s shape %v != %v", name, value.Shape, want)
		}
		return f32of(value.Values), nil
	}
	frames, width := fixture.Latents.Shape[0], fixture.Latents.Shape[1]
	codec, err := loadCodecDecoder(read, shapes, fixture.Config, width, fixtureSpeechProgram())
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Dims: Dims{LatentDim: width}, Codec: codec, EmbMean: fixture.Mean, EmbStd: fixture.Std}
	values := f32of(fixture.Latents.Values)
	whole, err := m.LatentsToPCM(t.Context(), LatentBatch{Frames: frames, Width: width, Values: values})
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "independent PocketTTS PCM", whole, fixture.PCM, tolSeanetPCM)
	for _, chunk := range []int{1, codec.window, frames} {
		t.Run(fmt.Sprint(chunk), func(t *testing.T) {
			t.Parallel()
			state := newCodecState(codec)
			var got []float32
			for start := 0; start < frames; start += chunk {
				end := min(start+chunk, frames)
				pcm, err := m.decodeLatents(t.Context(), state, LatentBatch{Values: values[start*width : end*width], Frames: end - start, Width: width})
				if err != nil {
					t.Fatal(err)
				}
				if len(pcm) != len(whole)/frames*(end-start) {
					t.Fatalf("chunk emitted %d samples", len(pcm))
				}
				got = append(got, pcm...)
			}
			if !slices.Equal(got, whole) {
				t.Fatal("chunk boundaries changed samples")
			}
			retained := func() int {
				count := 0
				for _, group := range [][][]float32{state.history, state.workspace, state.k, state.v} {
					for _, buffer := range group {
						count += cap(buffer)
					}
				}
				return count
			}
			bound := 0
			for repetition := range tensor.PairedExtent {
				for start := 0; start < frames; start += chunk {
					end := min(start+chunk, frames)
					if _, err := m.decodeLatents(t.Context(), state, LatentBatch{Values: values[start*width : end*width], Frames: end - start, Width: width}); err != nil {
						t.Fatal(err)
					}
				}
				if repetition != 0 && retained() != bound {
					t.Fatal("decoder history grew after overlap workspace was filled")
				}
				bound = retained()
			}
			if flushed, err := state.decode(t.Context(), nil, 0); err != nil || len(flushed) != 0 {
				t.Fatalf("final trim emitted tail: %v", err)
			}
			// A stopped decoder cannot resume after partially updating its history.
			ctx, cancel := context.WithCancelCause(t.Context())
			cause := errors.New("stop decoder")
			cancel(cause)
			one := LatentBatch{Values: values[:width], Frames: 1, Width: width}
			if pcm, err := m.decodeLatents(ctx, state, one); !errors.Is(err, cause) || pcm != nil {
				t.Fatalf("cancel: %v", err)
			}
			if _, err := m.decodeLatents(t.Context(), state, one); !errors.Is(err, cause) {
				t.Fatalf("resumed poisoned decoder: %v", err)
			}
		})
	}
}

func testStreamingFrameBoundaries(t *testing.T) {
	m := tinyJointModel(rand.New(rand.NewSource(1)))
	voice := &VoiceState{Frames: 1, K: make([][]float32, m.Dims.Layers), V: make([][]float32, m.Dims.Layers)}
	for i := range voice.K {
		voice.K[i] = make([]float32, m.Dims.DModel)
		voice.V[i] = make([]float32, m.Dims.DModel)
	}
	for _, test := range []struct {
		limit    int
		eos      float64
		complete bool
	}{
		{tensor.PairedExtent, math.Inf(-1), false},
		{tensor.TripleExtent, math.Inf(-1), true},
		{tensor.PairedExtent, math.Inf(1), false},
	} {
		calls := 0
		p := GenerateParams{MaxFrames: test.limit, FramesAfterEOS: tensor.PairedExtent, EOSThreshold: test.eos, NoiseAt: func(_ int, dst []float32) {
			calls++
			for i := range dst {
				dst[i] = float32(calls)
			}
		}}
		ids := []int{1}
		want, wantEOS, err := m.GenerateLatentsWithVoice(t.Context(), voice, ids, p)
		if err != nil {
			t.Fatal(err)
		}
		calls = 0
		state, err := m.promptVoice(t.Context(), voice, ids, p)
		if err != nil {
			t.Fatal(err)
		}
		var got []float32
		var logits []float64
		complete, err := m.walkFrames(t.Context(), state, p, func(frame []float32, logit float64) error {
			got = append(got, frame...)
			logits = append(logits, logit)
			return nil
		})
		if err != nil || complete != test.complete || calls != tensor.PairedExtent || !slices.Equal(got, want.Values) || !slices.Equal(logits, wantEOS) {
			t.Fatalf("stream boundaries complete=%v noise=%d err=%v", complete, calls, err)
		}
		for _, cancelContext := range []bool{false, true} {
			calls = 0
			state, err := m.promptVoice(t.Context(), voice, ids, p)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancelCause(t.Context())
			cause := errors.New("consumer stopped")
			complete, err := m.walkFrames(ctx, state, p, func([]float32, float64) error {
				if cancelContext {
					cancel(cause)
					return nil
				}
				return cause
			})
			cancel(cause)
			if complete || !errors.Is(err, cause) || calls != 1 {
				t.Fatalf("consumer stop complete=%v noise=%d err=%v", complete, calls, err)
			}
		}
	}
}

// speakFrom is Speak with the noise of each segment's generation given by the
// caller: a served request draws it from its seed, and a comparison with the
// model's own code hands in the noise that code drew.
func (s *Synthesizer) speakFrom(ctx context.Context, request DocumentRequest, noise func(generationPlan) frameNoise) (Audio, error) {
	return collectSpeech(func(emit func(Audio) error) (Audio, error) { return s.speakStream(ctx, request, noise, emit) })
}
