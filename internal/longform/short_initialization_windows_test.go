package longform

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
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
		if err != nil {
			t.Fatal(err)
		}
		runner, err := inference.OpenWithProgram(t.Context(), &program, inference.OpenOptions{})
		if err != nil {
			t.Fatal(err)
		}
		func() {
			defer runner.Close()
			if err := Warm(t.Context(), runner, corpus[:floors.PromptTokens], GuardContinuation); err != nil {
				t.Fatal(err)
			}
			if explicit {
				if err := Warm(t.Context(), runner, corpus[:floors.ShortPromptTokens], GuardContinuation); err != nil {
					t.Fatal(err)
				}
			}
			shape, err := Short(t.Context(), runner, corpus, floors, GuardContinuation)
			if err != nil {
				t.Fatal(err)
			}
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
