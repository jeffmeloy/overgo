package longform

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
)

// A committed run reads back by its weights location as the latest
// record, whatever the spelling of the path.
func TestPublishedRunIsTheLatestRecordOfItsLocation(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	weightsPath := filepath.Join(t.TempDir(), "weights.gguf")
	testcheck.NoError(t, os.WriteFile(weightsPath, []byte("weights fixture bytes"), 0o644))
	weights := testcheck.Must(os.Open(weightsPath))(t)
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	weights.Close()
	testcheck.NoError(t, err)
	commit := strings.Repeat("ab", 20)
	surface := strings.Repeat("5e", 32)

	floors := DeclaredFloors()
	failing := Result{
		ModelPath: weightsPath, ModelName: "fixture", Commit: commit, Surface: surface, PromptSource: "README.md",
		Measure: Measure{PromptTokens: 2000, OutputTokens: 256, PromptMilliseconds: 1000, DecodeMilliseconds: 6000,
			PromptTokensPerSecond: 400, DecodeTokensPerSecond: 40, DistinctFourGramRatio: 0.9, LongestRepeatedSpan: 3,
			Score: ContextScore{ScoreTokens: 128, LongContextNLL: 1.2, ShortContextNLL: 1.5, ContextGain: 0.3}},
		Short: ShortRates{PromptTokensPerSecond: 2000, DecodeTokensPerSecond: 40}, Floors: floors,
	}
	failing.Verdict = Judge(failing.Measure, failing.Short, floors)
	if failing.Verdict.Passed {
		t.Fatalf("fixture verdict passed: %+v", failing.Verdict)
	}
	claim, _, evidence, err := Claim(failing)
	testcheck.NoError(t, err)
	if claim.Capability != Capability || claim.Tier != runrecord.TierCapabilityMeasured || claim.Commit != commit ||
		claim.ContextTokens != 2000 || claim.WallNS != 7_000_000_000 || len(claim.Evidence) != 1 || claim.Evidence[0] != evidence {
		t.Fatalf("claim = %+v", claim)
	}
	if _, err := Publish(ctx, store, model, failing); err != nil {
		t.Fatal(err)
	}

	passing := failing
	passing.RealizedResidency = recipe.RealizedHostReference
	passing.Measure.PromptTokensPerSecond = 2400
	passing.Verdict = Judge(passing.Measure, passing.Short, floors)
	record := testcheck.Must(Publish(ctx, store, model, passing))(t)
	byLocation := LatestByLocation(ctx, store, 0)
	summary, found := byLocation[Key(strings.ToUpper(filepath.ToSlash(weightsPath)))]
	if !found || summary.Record != record || !summary.Result.Verdict.Passed || summary.Result.Measure.PromptTokensPerSecond != 2400 {
		t.Fatalf("latest = (%+v, %t)", summary, found)
	}
	if len(byLocation) != 1 {
		t.Fatalf("locations = %d, want 1", len(byLocation))
	}
}
