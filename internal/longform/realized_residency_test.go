package longform

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

func TestAdmissionRefusesMixedRealizedResidency(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	path := filepath.Join(t.TempDir(), "weights.gguf")
	if err := os.WriteFile(path, []byte("realized residency fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, _, identifyErr := artifact.Identify(artifact.KindModel, file)
	closeErr := file.Close()
	if err := errors.Join(identifyErr, closeErr); err != nil {
		t.Fatal(err)
	}
	surface := strings.Repeat("a", 64)
	result := Result{
		ModelPath: path, ModelName: "residency fixture", Commit: strings.Repeat("b", 40), Surface: surface,
		Inputs: Inputs{Model: model, CorpusDigest: strings.Repeat("c", 64), TokenDigest: strings.Repeat("d", 64), Protocol: RawContinuation},
		Floors: DeclaredFloors(), Verdict: Verdict{Passed: true},
		Measure: Measure{PromptTokens: 1, PromptMilliseconds: 1},
		Shape:   ShortShape{Measure: Measure{PromptTokensPerSecond: 100, DecodeTokensPerSecond: 100}},
	}
	if _, err := Publish(t.Context(), store, model, result); err != nil {
		t.Fatal(err)
	}
	if err := Admit(t.Context(), store, path, surface); !errors.Is(err, ErrRefused) || !strings.Contains(err.Error(), "lacks realized residency") {
		t.Fatalf("legacy residency admitted: %v", err)
	}
	result.RealizedResidency = recipe.RealizedDeviceNative
	if _, err := Publish(t.Context(), store, model, result); err != nil {
		t.Fatal(err)
	}
	if err := Admit(t.Context(), store, path, surface); err != nil {
		t.Fatalf("observed residency refused: %v", err)
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
