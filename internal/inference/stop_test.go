package inference

import (
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

func TestStopSequenceValidationAndMatching(t *testing.T) {
	policy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		t.Fatalf("runtime policy: found=%t err=%v", found, err)
	}
	limit := policy.Serving.Limits.StopSequences
	if err := validateStopSequences([]string{"END", "stop"}, limit); err != nil {
		t.Fatal(err)
	}
	if err := validateStopSequences([]string{""}, limit); err == nil {
		t.Fatal("empty stop sequence was accepted")
	}
	if err := validateStopSequences(make([]string, limit+1), limit); err == nil {
		t.Fatal("oversized stop sequence set was accepted")
	}
	if !matchesStopSequence("prefix END suffix", []string{"END"}) {
		t.Fatal("stop sequence was not matched")
	}
	if matchesStopSequence("prefix EN", []string{"END"}) {
		t.Fatal("partial stop sequence matched")
	}
}
