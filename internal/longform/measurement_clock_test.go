package longform

import (
	"testing"
	"time"

	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestGenerationMeasurementClock(t *testing.T) {
	path := testutil.HermeticLlamaGGUF(t, 16)
	program, err := servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithProgram(t.Context(), &program, inference.OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	before, err := processmeasure.Counter()
	if err != nil {
		t.Fatal(err)
	}
	generated, err := Run(t.Context(), runner, []tokenizer.TokenID{1, 4}, 3, GuardContinuation)
	after, clockErr := processmeasure.Counter()
	if err != nil || clockErr != nil {
		t.Fatalf("generation: %v; clock: %v", err, clockErr)
	}
	m := generated.Measure
	wallMS := float64(after-before) / float64(time.Millisecond)
	if m.PromptTokens != 2 || m.OutputTokens != 3 || m.PromptMilliseconds <= 0 || m.DecodeMilliseconds <= 0 || m.PromptMilliseconds+m.DecodeMilliseconds > wallMS || m.PromptTokensPerSecond <= 0 || m.DecodeTokensPerSecond <= 0 {
		t.Fatalf("invalid measured generation: %+v wall=%gms", m, wallMS)
	}
}
