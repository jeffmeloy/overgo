package speechsynth

import (
	"slices"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
)

const (
	synthesisTextFixture   = "hello"
	synthesisFramesFixture = 2
	synthesisSeedFixture   = 7
	synthesisRateFixture   = 24000
)

var (
	synthesisPCMFixture    = []float32{0.25, -0.5}
	synthesisTokenFixture  = []int{3, 5}
	synthesisLatentFixture = []float32{0.5, 0.75}
)

type runtimeFixture struct {
	t       *testing.T
	request SynthesisRequest
	plan    generationPlan
	latents LatentBatch
	audio   Audio
}

func (f runtimeFixture) tokenize(request SynthesisRequest) (generationPlan, error) {
	if request != f.request {
		f.t.Fatalf("request = %+v", request)
	}
	return f.plan, nil
}

func (f runtimeFixture) generate(plan generationPlan) (LatentBatch, error) {
	if !slices.Equal(plan.tokens, f.plan.tokens) || plan.maxFrames != f.plan.maxFrames || plan.seed != f.plan.seed {
		f.t.Fatalf("plan = %+v", plan)
	}
	return f.latents, nil
}

func (f runtimeFixture) decode(latents LatentBatch) (Audio, error) {
	if !slices.Equal(latents.Values, f.latents.Values) || latents.Frames != f.latents.Frames || latents.Width != f.latents.Width {
		f.t.Fatalf("latents = %+v", latents)
	}
	return f.audio, nil
}

func TestRegisteredRuntimeExecutesSpeechProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "speech-model", modelrecipe.SpeechDefinition)
	request := SynthesisRequest{Text: synthesisTextFixture, MaxFrames: synthesisFramesFixture, Seed: synthesisSeedFixture}
	plan := generationPlan{tokens: synthesisTokenFixture, maxFrames: synthesisFramesFixture, seed: synthesisSeedFixture}
	latents := LatentBatch{Values: synthesisLatentFixture, Frames: 1, Width: len(synthesisLatentFixture)}
	audio := Audio{PCM: synthesisPCMFixture, SampleRate: synthesisRateFixture}
	if err := registerRuntime(fixture.Runtime, fixture.Model, runtimeFixture{
		t: t, request: request, plan: plan, latents: latents, audio: audio,
	}); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.ExecuteScalar("speech/runtime", request)
	if err != nil {
		t.Fatal(err)
	}
	datum, one := result.Outputs["audio"].Single()
	got, typed := datum.Value.(Audio)
	if !one || !typed || got.SampleRate != synthesisRateFixture || !slices.Equal(got.PCM, synthesisPCMFixture) || !result.Commit.Valid() {
		t.Fatalf("audio = (%+v, %v, %v), commit=%v", got, one, typed, result.Commit)
	}
}
