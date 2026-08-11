package seq2seq

import (
	"slices"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
)

const generationLimitFixture = 2

var (
	generationSourceFixture = []int{3, 5}
	generationOutputFixture = []int{8, 13}
	generationMemoryFixture = []float32{0.25, 0.5}
)

type selectFunc func() ([]int, error)

func (f selectFunc) Select() ([]int, error) { return f() }

type runtimeFixture struct {
	t       *testing.T
	request GenerateRequest
	encoded EncodedRequest
	output  []int
}

func (f runtimeFixture) EncodeRequest(request GenerateRequest) (EncodedRequest, error) {
	if !slices.Equal(request.Source, f.request.Source) || request.MaxTokens != f.request.MaxTokens {
		f.t.Fatalf("request = %+v", request)
	}
	return f.encoded, nil
}

func (f runtimeFixture) PrepareGeneration(encoded EncodedRequest) (TokenSelector, error) {
	if !slices.Equal(encoded.Memory, f.encoded.Memory) ||
		encoded.SourceRows != f.encoded.SourceRows || encoded.MaxTokens != f.encoded.MaxTokens {
		f.t.Fatalf("encoded = %+v", encoded)
	}
	return selectFunc(func() ([]int, error) { return f.output, nil }), nil
}

func TestRegisteredRuntimeExecutesSeq2SeqProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "seq2seq-model", modelrecipe.Seq2SeqDefinition)
	request := GenerateRequest{Source: generationSourceFixture, MaxTokens: generationLimitFixture}
	encoded := EncodedRequest{Memory: generationMemoryFixture, SourceRows: len(generationSourceFixture), MaxTokens: generationLimitFixture}
	if err := registerRuntime(fixture.Runtime, fixture.Model, runtimeFixture{
		t: t, request: request, encoded: encoded, output: generationOutputFixture,
	}); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.ExecuteScalar("seq2seq/runtime", request)
	if err != nil {
		t.Fatal(err)
	}
	datum, one := result.Outputs["tokens"].Single()
	got, typed := datum.Value.([]int)
	if !one || !typed || !slices.Equal(got, generationOutputFixture) || !result.Commit.Valid() {
		t.Fatalf("tokens = (%v, %v, %v), commit=%v", got, one, typed, result.Commit)
	}
}
