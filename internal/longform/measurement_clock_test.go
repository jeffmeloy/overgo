package longform

import (
	"testing"
	"time"

	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestGenerationMeasurementClock(t *testing.T) {
	path := testutil.HermeticLlamaGGUF(t, 16)
	program := testcheck.Must(servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference))(t)
	runner := testcheck.Must(inference.OpenWithProgram(t.Context(), &program, inference.OpenOptions{}))(t)
	defer runner.Close()
	before := testcheck.Must(processmeasure.Counter())(t)
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
