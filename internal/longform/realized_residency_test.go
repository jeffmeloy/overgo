package longform

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

func TestComparisonRefusesMixedRealizedResidency(t *testing.T) {
	model, _, err := artifact.Identify(artifact.KindModel, strings.NewReader("realized residency fixture"))
	if err != nil {
		t.Fatal(err)
	}
	result := Result{
		ModelName: "residency fixture", Commit: strings.Repeat("b", 40), Surface: strings.Repeat("a", 64),
		Inputs: Inputs{Model: model, CorpusDigest: strings.Repeat("c", 64), TokenDigest: strings.Repeat("d", 64), Protocol: RawContinuation},
		Floors: DeclaredFloors(), Verdict: Verdict{Passed: true},
		Measure:           Measure{PromptTokens: 1, PromptMilliseconds: 1},
		Shape:             ShortShape{Measure: Measure{PromptTokensPerSecond: 100, DecodeTokensPerSecond: 100}},
		RealizedResidency: recipe.RealizedDeviceNative,
	}
	if verdict := Compare(result, result, result.Floors, 0); !verdict.Passed {
		t.Fatalf("matched residency refused: %+v", verdict)
	}
	fallback := result
	fallback.RealizedResidency = recipe.RealizedOOMStreamed
	if verdict := Compare(result, fallback, result.Floors, 0); verdict.Passed ||
		len(verdict.Reasons) != 1 || !strings.Contains(verdict.Reasons[0], "realized residency") {
		t.Fatalf("mixed residency compared: %+v", verdict)
	}
	legacy := result
	legacy.RealizedResidency = ""
	if verdict := Compare(legacy, result, result.Floors, 0); verdict.Passed {
		t.Fatalf("legacy comparison passed: %+v", verdict)
	}
}
