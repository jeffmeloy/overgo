package seq2seq

import (
	"testing"

	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
)

const generationLimitFixture = 2

var generationMemoryFixture = []float32{0.25, 0.5}

type textSelectFunc func() (string, error)

func (f textSelectFunc) selectText() (string, error) { return f() }

type runtimeFixture struct {
	t       *testing.T
	request GenerateRequest
	encoded encodedRequest
	output  string
}

func (f runtimeFixture) encodeRequest(request GenerateRequest) (encodedRequest, error) {
	if request != f.request {
		f.t.Fatalf("request = %+v", request)
	}
	return f.encoded, nil
}

func (f runtimeFixture) prepareGeneration(encoded encodedRequest) (textSelector, error) {
	if encoded.sourceRows != f.encoded.sourceRows || encoded.maxTokens != f.encoded.maxTokens || len(encoded.memory) != len(f.encoded.memory) {
		f.t.Fatalf("encoded = %+v", encoded)
	}
	return textSelectFunc(func() (string, error) { return f.output, nil }), nil
}

func TestRegisteredRuntimeExecutesSeq2SeqProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "seq2seq-model", recipe.TaskSeq2Seq)
	request := GenerateRequest{Text: "weather in San Francisco", MaxTokens: generationLimitFixture}
	encoded := encodedRequest{memory: generationMemoryFixture, sourceRows: 4, maxTokens: generationLimitFixture}
	if err := registerRuntime(fixture.Runtime, fixture.Model, runtimeFixture{
		t: t, request: request, encoded: encoded, output: "get_weather",
	}); err != nil {
		t.Fatal(err)
	}
	got := modelrecipetest.MustExecuteScalar[string](t, fixture, "seq2seq/runtime", request)
	if got != "get_weather" {
		t.Fatalf("text = %q", got)
	}
}
