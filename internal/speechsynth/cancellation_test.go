package speechsynth

import (
	"context"
	"errors"
	"math"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
	"slices"
	"testing"
)

func TestSpeechFrameCancellation(t *testing.T) {
	m := loadArtifactModel(t)
	g6 := loadFixture[g6Golden](t, "g6_backbone.json")
	g7 := loadFixture[g7Golden](t, "g7_generation.json")
	g1 := loadFixture[struct {
		IDs []int `json:"ids"`
	}](t, "g1_tokenizer.json")
	path, err := ResolveVoicePath(artifactDir(t), "alba")
	if err != nil {
		t.Fatal(err)
	}
	voice, err := LoadVoiceState(path, m.Dims)
	if err != nil {
		t.Fatal(err)
	}
	entries := map[string]func(context.Context, GenerateParams) (LatentBatch, []float64, error){
		"conditioning": func(ctx context.Context, p GenerateParams) (LatentBatch, []float64, error) {
			return m.GenerateLatents(ctx, f32of(g6.TransformerCalls[0].In.Values), g6.TransformerCalls[0].In.Shape[1], g1.IDs, p)
		},
		"stored voice": func(ctx context.Context, p GenerateParams) (LatentBatch, []float64, error) {
			return m.GenerateLatentsWithVoice(ctx, voice, g1.IDs, p)
		},
	}
	for name, generate := range entries {
		t.Run(name, func(t *testing.T) {
			normal := GenerateParams{MaxFrames: g7.NQuantizerCalls, EOSThreshold: math.Inf(1), NoiseAt: func(step int, dst []float32) { copy(dst, f32of(g7.FlowCalls[2+step].Noise.Values)) }}
			baseline, baselineEOS, err := generate(t.Context(), normal)
			if err != nil {
				t.Fatal(err)
			}
			for _, before := range []bool{true, false} {
				ctx, cancel := context.WithCancelCause(t.Context())
				cause := errors.New("stop this speech request")
				calls := 0
				if before {
					cancel(cause)
				}
				latents, eos, err := generate(ctx, GenerateParams{MaxFrames: g7.NQuantizerCalls, EOSThreshold: math.Inf(1), NoiseAt: func(step int, dst []float32) { calls++; cancel(cause) }})
				cancel(cause)
				wantCalls := 1
				if before {
					wantCalls = 0
				}
				if !errors.Is(err, cause) || calls != wantCalls || len(latents.Values) != 0 || len(eos) != 0 {
					t.Fatalf("cancel before=%v calls=%d frames=%d error=%v", before, calls, latents.Frames, err)
				}
			}
			recovered, recoveredEOS, err := generate(t.Context(), normal)
			if err != nil {
				t.Fatal(err)
			}
			if recovered.Frames != baseline.Frames || !slices.Equal(recovered.Values, baseline.Values) || !slices.Equal(recoveredEOS, baselineEOS) {
				t.Fatal("cancellation changed a later generation using the same model and voice state")
			}
		})
	}
}

func TestSpeechCodecCancellation(t *testing.T) {
	m := loadArtifactModel(t)
	ctx, cancel := context.WithCancelCause(t.Context())
	cause := errors.New("stop decoding speech")
	cancel(cause)
	if pcm, err := m.LatentsToPCM(ctx, LatentBatch{Values: make([]float32, m.Dims.LatentDim), Frames: 1, Width: m.Dims.LatentDim}); !errors.Is(err, cause) || pcm != nil {
		t.Fatalf("canceled decoding returned %d samples: %v", len(pcm), err)
	}
	if pcm, err := m.LatentsToPCM(nil, LatentBatch{}); err == nil || pcm != nil {
		t.Fatal("nil decoding context accepted")
	}
}

func TestSpeechPipelineCancellation(t *testing.T) {
	synthesizer, err := LoadSynthesizer(artifactDir(t))
	if err != nil {
		t.Fatal(err)
	}
	capability := modelrecipetest.NewCapability(t, "speech-cancellation-fixture", recipe.TaskSpeech)
	if err := RegisterRuntime(capability.Runtime, capability.Model, synthesizer); err != nil {
		t.Fatal(err)
	}
	definition := capability.Program.Definition()
	input := definition.Inputs[0]
	request := SynthesisRequest{Text: "Green always green.", Voice: "alba", MaxFrames: 2, Seed: 7}
	for _, key := range []string{"stopped", "retry"} {
		t.Run(key, func(t *testing.T) {
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(context.Canceled)
			if key == "stopped" {
				cancel(context.Canceled)
			}
			execution, err := workflowruntime.ExecutionID(definition.ID, key)
			if err != nil {
				t.Fatal(err)
			}
			result, err := capability.Runtime.ExecuteProgram(ctx, key, execution, nil, capability.Program, map[recipe.PortName]workflowruntime.Value{
				input.Name: {Kind: input.Data, Items: []workflowruntime.Datum{{Value: request}}},
			})
			if key == "stopped" {
				if !errors.Is(err, context.Canceled) || len(result.Outputs) != 0 || result.Run.Outcome != runrecord.OutcomeCancelled || !result.Commit.Valid() {
					t.Fatalf("cancelled result: %+v err=%v", result, err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				audio := modelrecipetest.Output[Audio](t, result, definition.Outputs[0].Name)
				if result.Run.Outcome != runrecord.OutcomeSucceeded || len(audio.PCM) == 0 || audio.SampleRate <= 0 {
					t.Fatal("same runtime failed to recover after cancellation")
				}
			}
		})
	}
}
