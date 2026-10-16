package speechsynth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

type codecFixture struct {
	Config              artifactConfig          `json:"config"`
	Tensors             map[string]goldenTensor `json:"tensors"`
	Mean, Std           []float32
	Latents             goldenTensor
	PCM                 []float64
	VoiceInput          []float32 `json:"voice_input"`
	Conditioning        goldenTensor
	ConditioningWithBOS []float64 `json:"conditioning_with_bos"`
	Resampled           []struct {
		From, To int
		Values   []float64
	}
}

func readCodecFixture(t *testing.T) (codecFixture, map[string][]int, func(string, ...int) ([]float32, error)) {
	t.Helper()
	var fixture codecFixture
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
	return fixture, shapes, read
}

func TestMimiStreamingDecode(t *testing.T) {
	t.Run("generation boundaries", testStreamingFrameBoundaries)
	fixture, shapes, read := readCodecFixture(t)
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

func TestVoicePreparation(t *testing.T) {
	fixture, shapes, read := readCodecFixture(t)
	if len(fixture.Conditioning.Shape) != tensor.PairedExtent {
		t.Fatal("independent voice reference is absent")
	}
	encoder, err := loadCodecEncoder(read, shapes, fixture.Config, fixture.Conditioning.Shape[1], fixtureSpeechProgram())
	if err != nil {
		t.Fatal(err)
	}
	conditioning, padding, err := encoder.encode(t.Context(), fixture.VoiceInput)
	if err != nil {
		t.Fatal(err)
	}
	if padding != 1 || len(conditioning) != len(fixture.Conditioning.Values) {
		t.Fatalf("padding=%d conditioning=%d", padding, len(conditioning))
	}
	requireWithin(t, "independent encoder and speaker projection", conditioning, fixture.Conditioning.Values, tolTransformer)
	withBOS := fixture.Config
	withBOS.FlowLM.InsertBOSBeforeVoice = true
	withEncoder, err := loadCodecEncoder(read, shapes, withBOS, fixture.Conditioning.Shape[1], fixtureSpeechProgram())
	if err != nil {
		t.Fatal(err)
	}
	prefixed, _, err := withEncoder.encode(t.Context(), fixture.VoiceInput)
	if err != nil {
		t.Fatal(err)
	}
	requireWithin(t, "independent configured voice prefix", prefixed, fixture.ConditioningWithBOS, tolTransformer)
	for _, reference := range fixture.Resampled {
		values, _, err := resampleVoice(t.Context(), fixture.VoiceInput, reference.From, reference.To, math.MaxInt64)
		if err != nil {
			t.Fatal(err)
		}
		requireWithin(t, "independent resampling", values, reference.Values, tolUpsample)
	}
	if err := encoder.admit(len(fixture.VoiceInput), Dims{DModel: 8, Layers: 2, FF: 16, Heads: 2}, 1); err == nil {
		t.Fatal("unfunded encoder admitted")
	}
	for _, rates := range [][3]int{{16000, 24000, 1}, {1, math.MaxInt, math.MaxInt}, {0, 24000, math.MaxInt}} {
		if _, _, err := resampleVoice(t.Context(), fixture.VoiceInput, rates[0], rates[1], int64(rates[2])); err == nil {
			t.Fatalf("invalid resampling admitted: %v", rates)
		}
	}
	for _, name := range []string{"cancelled", "empty", "nonfinite", "missing weights", "disabled encoder"} {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			samples := slices.Clone(fixture.VoiceInput)
			switch name {
			case "cancelled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "empty":
				samples = nil
			case "nonfinite":
				samples[0] = float32(math.NaN())
			case "missing weights", "disabled encoder":
				broken := func(key string, want ...int) ([]float32, error) {
					if key == "mimi.encoder.model.0.conv.weight" {
						if name == "missing weights" {
							return nil, errors.New("missing encoder")
						}
						values, err := read(key, want...)
						clear(values)
						return values, err
					}
					return read(key, want...)
				}
				if _, err := loadCodecEncoder(broken, shapes, fixture.Config, fixture.Conditioning.Shape[1], fixtureSpeechProgram()); err == nil {
					t.Fatal("unavailable encoder accepted")
				}
				return
			}
			if values, _, err := encoder.encode(ctx, samples); err == nil || values != nil {
				t.Fatalf("invalid input produced conditioning: %v", err)
			}
		})
	}
	// The fixture crosses the encoder attention window and ends before a full codec frame.
	// A separate zero-padded request has the same conditioning but reports no added padding.
	complete := append(slices.Clone(fixture.VoiceInput), 0)
	again, extra, err := encoder.encode(t.Context(), complete)
	if err != nil || extra != 0 || !slices.Equal(again, conditioning) {
		t.Fatalf("complete frame padding changed conditioning: %v", err)
	}
	testPreparedVoiceBinding(t, fixture, shapes)
}

func testPreparedVoiceBinding(t *testing.T, fixture codecFixture, shapes map[string][]int) {
	t.Helper()
	directory := t.TempDir()
	fixture.Config.FlowLM.Transformer.NumHeads = tensor.PairedExtent
	fixture.Config.FlowLM.Transformer.MaxPeriod = fixture.Config.Mimi.Transformer.MaxPeriod
	config, err := json.Marshal(fixture.Config)
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{configFileName: config, "tokenizer.model": []byte("unused tokenizer inventory fixture")} {
		if err := os.WriteFile(filepath.Join(directory, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	tensors := make(map[string][]float32)
	for name, value := range fixture.Tensors {
		tensors[name] = f32of(value.Values)
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), tensors, shapes, nil); err != nil {
		t.Fatal(err)
	}
	inventory, err := Inventory(directory)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSpeech, inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch, err := inventory.Batch("fixture-model")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "fixture-recipe", definition); err != nil {
		t.Fatal(err)
	}
	wav, err := media.EncodeWAVPCM16(fixture.VoiceInput, fixture.Config.Mimi.SampleRate)
	if err != nil {
		t.Fatal(err)
	}
	recording, err := (artifact.DocumentContract{Kind: artifact.KindFile, MediaType: media.WAVMediaType, Schema: "fixture/recording"}).ContentBytes(wav)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.Publish(t.Context(), store, artifact.Batch{Key: "fixture-recording", Contents: []artifact.Content{recording}}); err != nil {
		t.Fatal(err)
	}
	model := tinyBackboneModel(rand.New(rand.NewSource(1)), fixture.Conditioning.Shape[1], 2, 16, 2)
	model.Codec = &CodecDecoder{SampleRate: fixture.Config.Mimi.SampleRate}
	synth := &Synthesizer{model: model, tokenizer: &Unigram{}, directory: directory, boundRecipe: definition.ID, voices: map[string]*VoiceState{}}
	request := SynthesisRequest{Text: "a", ReferenceAudio: recording.Descriptor.ID, ReferenceEndSample: len(fixture.VoiceInput), VoiceMemoryBytes: artifact.MaxContentBytes}
	id, err := synth.BindVoice(t.Context(), store, definition, request)
	if err != nil {
		t.Fatal(err)
	}
	voice, err := synth.voiceState(request.voiceKey())
	if err != nil || voice.Frames != fixture.Conditioning.Shape[0] {
		t.Fatalf("prepared frames: %v %v", voice, err)
	}
	head, sequence := store.Head()
	// Reuse avoids source decoding and encoder loading while inventory identity is rechecked.
	request.VoiceMemoryBytes = 1
	again, err := synth.BindVoice(t.Context(), store, definition, request)
	after, next := store.Head()
	if err != nil || again != id || head != after || sequence != next {
		t.Fatalf("reuse changed state: %s %v", again, err)
	}
	prepared := SynthesisRequest{Text: request.Text, Voice: " ", PreparedVoice: id}
	clear(synth.voices)
	if _, err := synth.BindVoice(t.Context(), store, definition, prepared); err != nil {
		t.Fatal(err)
	}
	restored, err := synth.voiceState(prepared.voiceKey())
	if err != nil || !reflect.DeepEqual(restored, voice) {
		t.Fatalf("restored voice differs: %v", err)
	}
	for _, name := range []string{"conflict", "range", "budget", "cancel", "recipe", "corrupt state"} {
		t.Run(name, func(t *testing.T) {
			candidate, chosen, ctx := request, definition, t.Context()
			candidate.VoiceMemoryBytes = artifact.MaxContentBytes
			candidate.ReferenceStartSample = 1
			switch name {
			case "conflict":
				candidate.Voice = "alba"
			case "range":
				candidate.ReferenceEndSample++
			case "budget":
				candidate.VoiceMemoryBytes = 1
			case "cancel":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "recipe":
				candidate = prepared
				checkpoint, err := artifact.IdentifyBytes(artifact.KindCheckpoint, []byte("another trained surface"))
				if err != nil {
					t.Fatal(err)
				}
				chosen, err = modelrecipe.AdaptedSpeechDefinition(definition, checkpoint)
				if err != nil {
					t.Fatal(err)
				}
				synth.boundRecipe = chosen.ID
				defer func() { synth.boundRecipe = definition.ID }()
			case "corrupt state":
				content, err := artifact.RequireTypedContent(ctx, store, id)
				if err != nil {
					t.Fatal(err)
				}
				var broken preparedVoice
				if err := json.Unmarshal(content.Data, &broken); err != nil {
					t.Fatal(err)
				}
				broken.State.Frames++
				content, err = artifact.JSONContent(preparedVoiceContract, broken)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := artifact.Publish(ctx, store, artifact.Batch{Key: "broken-voice", Contents: []artifact.Content{content}}); err != nil {
					t.Fatal(err)
				}
				candidate = SynthesisRequest{Text: request.Text, PreparedVoice: content.Descriptor.ID}
			}
			if _, err := synth.BindVoice(ctx, store, chosen, candidate); err == nil {
				t.Fatal("invalid voice accepted")
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
