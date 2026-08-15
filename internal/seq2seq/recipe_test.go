package seq2seq

import (
	"testing"

	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
)

func generateThroughRecipe(t testing.TB, generator *Generator, request GenerateRequest) (string, error) {
	t.Helper()
	fixture := modelrecipetest.NewCapability(t, "seq2seq-model", recipe.TaskSeq2Seq)
	if err := RegisterRuntime(fixture.Runtime, fixture.Model, generator); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.ExecuteScalar("seq2seq/runtime", request)
	if err != nil {
		return "", err
	}
	return modelrecipetest.Output[string](t, result, fixture.Program.Definition().Outputs[0].Name), nil
}
