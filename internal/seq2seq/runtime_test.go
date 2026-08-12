package seq2seq

import (
	"slices"
	"testing"

	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
)

const generationLimitFixture = 2

var (
	generationSourceFixture = []int{3, 5}
	generationOutputFixture = []int{8, 13}
	generationMemoryFixture = []float32{0.25, 0.5}
)

type selectFunc func() ([]int, error)

func (f selectFunc) selectTokens() ([]int, error) { return f() }

type runtimeFixture struct {
	t       *testing.T
	request GenerateRequest
	encoded encodedRequest
	output  []int
}

func (f runtimeFixture) encodeRequest(request GenerateRequest) (encodedRequest, error) {
	if !slices.Equal(request.Source, f.request.Source) || request.MaxTokens != f.request.MaxTokens {
		f.t.Fatalf("request = %+v", request)
	}
	return f.encoded, nil
}

func (f runtimeFixture) prepareGeneration(encoded encodedRequest) (tokenSelector, error) {
	if !slices.Equal(encoded.memory, f.encoded.memory) ||
		encoded.sourceRows != f.encoded.sourceRows || encoded.maxTokens != f.encoded.maxTokens {
		f.t.Fatalf("encoded = %+v", encoded)
	}
	return selectFunc(func() ([]int, error) { return f.output, nil }), nil
}

func TestRegisteredRuntimeExecutesSeq2SeqProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "seq2seq-model", recipe.TaskSeq2Seq)
	request := GenerateRequest{Source: generationSourceFixture, MaxTokens: generationLimitFixture}
	encoded := encodedRequest{memory: generationMemoryFixture, sourceRows: len(generationSourceFixture), maxTokens: generationLimitFixture}
	if err := registerRuntime(fixture.Runtime, fixture.Model, runtimeFixture{
		t: t, request: request, encoded: encoded, output: generationOutputFixture,
	}); err != nil {
		t.Fatal(err)
	}
	got := modelrecipetest.MustExecuteScalar[[]int](t, fixture, "seq2seq/runtime", request)
	if !slices.Equal(got, generationOutputFixture) {
		t.Fatalf("tokens = %v", got)
	}
}
