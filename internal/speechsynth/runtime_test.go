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

var synthesisPCMFixture = []float32{0.25, -0.5}

type synthesisFunc func(SynthesisRequest) (Audio, error)

func (f synthesisFunc) Synthesize(request SynthesisRequest) (Audio, error) { return f(request) }

func TestRegisteredRuntimeExecutesSpeechProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "speech-model", modelrecipe.SpeechDefinition)
	request := SynthesisRequest{Text: synthesisTextFixture, MaxFrames: synthesisFramesFixture, Seed: synthesisSeedFixture}
	if err := registerRuntime(fixture.Runtime, fixture.Model, synthesisFunc(func(got SynthesisRequest) (Audio, error) {
		if got != request {
			t.Fatalf("request = %+v", got)
		}
		return Audio{PCM: synthesisPCMFixture, SampleRate: synthesisRateFixture}, nil
	})); err != nil {
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
