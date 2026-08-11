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
)

type generateFunc func([]int, int) ([]int, error)

func (f generateFunc) Generate(source []int, limit int) ([]int, error) { return f(source, limit) }

func TestRegisteredRuntimeExecutesSeq2SeqProgram(t *testing.T) {
	fixture := modelrecipetest.NewCapability(t, "seq2seq-model", modelrecipe.Seq2SeqDefinition)
	if err := registerRuntime(fixture.Runtime, fixture.Model, generateFunc(func(source []int, limit int) ([]int, error) {
		if !slices.Equal(source, generationSourceFixture) || limit != generationLimitFixture {
			t.Fatalf("request = (%v, %d)", source, limit)
		}
		return generationOutputFixture, nil
	})); err != nil {
		t.Fatal(err)
	}
	request := GenerateRequest{Source: generationSourceFixture, MaxTokens: generationLimitFixture}
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
