package longform

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// A committed run reads back by its weights location as the latest
// record, and the suite admission turns the record into the three
// refusals a pass can meet: no record, a record from other code, and a
// failed verdict; a passing record on this commit admits.
func TestPublishedRunAdmitsOrRefusesSuitePasses(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	weightsPath := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(weightsPath, []byte("weights fixture bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	weights, err := os.Open(weightsPath)
	if err != nil {
		t.Fatal(err)
	}
	model, _, err := artifact.Identify(artifact.KindModel, weights)
	weights.Close()
	if err != nil {
		t.Fatal(err)
	}
	commit := strings.Repeat("ab", 20)
	surface := strings.Repeat("5e", 32)
	if err := Admit(ctx, store, weightsPath, surface); !errors.Is(err, ErrRefused) || !strings.Contains(err.Error(), "no long-form record") {
		t.Fatalf("admission without a record = %v", err)
	}

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
	if err != nil {
		t.Fatal(err)
	}
	if claim.Capability != Capability || claim.Tier != runrecord.TierCapabilityMeasured || claim.Commit != commit ||
		claim.ContextTokens != 2000 || claim.WallNS != 7_000_000_000 || len(claim.Evidence) != 1 || claim.Evidence[0] != evidence {
		t.Fatalf("claim = %+v", claim)
	}
	if _, err := Publish(ctx, store, model, failing); err != nil {
		t.Fatal(err)
	}
	if err := Admit(ctx, store, weightsPath, surface); !errors.Is(err, ErrRefused) || !strings.Contains(err.Error(), "failed its floors") {
		t.Fatalf("admission on a failed verdict = %v", err)
	}

	passing := failing
	passing.Measure.PromptTokensPerSecond = 2400
	passing.Verdict = Judge(passing.Measure, passing.Short, floors)
	record, err := Publish(ctx, store, model, passing)
	if err != nil {
		t.Fatal(err)
	}
	summary, found := Latest(ctx, store, strings.ToUpper(filepath.ToSlash(weightsPath)))
	if !found || summary.Record != record || !summary.Result.Verdict.Passed || summary.Result.Measure.PromptTokensPerSecond != 2400 {
		t.Fatalf("latest = (%+v, %t)", summary, found)
	}
	if err := Admit(ctx, store, weightsPath, surface); err != nil {
		t.Fatalf("admission on a passing record = %v", err)
	}
	if err := Admit(ctx, store, weightsPath, strings.Repeat("6f", 32)); !errors.Is(err, ErrRefused) || !strings.Contains(err.Error(), "measured inference surface") {
		t.Fatalf("admission on another surface = %v", err)
	}
	if byLocation := LatestByLocation(ctx, store, 0); len(byLocation) != 1 {
		t.Fatalf("locations = %d, want 1", len(byLocation))
	}
}
