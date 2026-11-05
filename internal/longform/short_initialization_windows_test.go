package longform

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// Short must exclude shape-specific graph initialization from measured work.
func TestShortInitializesMeasuredShape(t *testing.T) {
	cudatest.Require(t)
	floors := DeclaredFloors()
	path := testutil.HermeticLlamaGGUF(t, uint32(floors.PromptTokens+floors.OutputTokens))
	corpus := make([]tokenizer.TokenID, floors.PromptTokens+floors.ScoreTokens)
	for i := range corpus {
		corpus[i] = 4
	}
	var control ShortShape
	for _, explicit := range []bool{true, false} {
		program, err := servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceF32)
		testcheck.NoError(t, err)
		runner, err := inference.OpenWithProgram(t.Context(), &program, inference.OpenOptions{})
		testcheck.NoError(t, err)
		func() {
			defer runner.Close()
			testcheck.NoError(t, Warm(t.Context(), runner, corpus[:floors.PromptTokens], GuardContinuation))
			if explicit {
				testcheck.NoError(t, Warm(t.Context(), runner, corpus[:floors.ShortPromptTokens], GuardContinuation))
			}
			shape, err := Short(t.Context(), runner, corpus, floors, GuardContinuation)
			testcheck.NoError(t, err)
			work := shape.Measure.Execution
			if shape.WarmupOutputTokens != WarmupOutputTokens || work.GraphLaunchesPerToken == 0 || work.GraphInstantiationsPerToken != 0 {
				t.Fatalf("explicit=%t measured graph initialization: %+v", explicit, work)
			}
			if explicit {
				control = shape
				return
			}
			if !slices.Equal(shape.OutputIDs, control.OutputIDs) || shape.NLL != control.NLL || shape.Measure.Memory.CurrentBytes != control.Measure.Memory.CurrentBytes || shape.Measure.Memory.PeakBytes != control.Measure.Memory.PeakBytes {
				t.Fatal("initialization changed tokens, NLL or owned allocation bounds")
			}
		}()
	}
}
