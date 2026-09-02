package evaluation

import (
	"crypto/sha256"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestReferenceScoresDeclareMergeAndLoad(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model, err := artifact.NewID(artifact.KindModel, sha256.Sum256([]byte("reference model")))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	if references, err := LoadReferenceScores(ctx, store, model); err != nil || len(references) != 0 {
		t.Fatalf("undeclared model = %v, %v", references, err)
	}
	first := ReferenceScore{Suite: "store/mmlu", Metric: "accuracy", Value: 0.475, Protocol: "5-shot", Source: "blog"}
	if _, err := DeclareReferenceScores(ctx, store, model, []ReferenceScore{first}); err != nil {
		t.Fatal(err)
	}
	replaced := first
	replaced.Value = 0.48
	other := ReferenceScore{Suite: "store/bbh", Metric: "accuracy", Value: 0.203, Protocol: "3-shot", Source: "blog"}
	if _, err := DeclareReferenceScores(ctx, store, model, []ReferenceScore{other, replaced}); err != nil {
		t.Fatal(err)
	}
	references, err := LoadReferenceScores(ctx, store, model)
	if err != nil || len(references) != 2 {
		t.Fatalf("references = %v, %v", references, err)
	}
	if references[0] != other || references[1] != replaced {
		t.Fatalf("merged order/values = %v", references)
	}
	invalid := ReferenceScore{Suite: "store/mmlu", Metric: "", Value: 1, Protocol: "p", Source: "s"}
	if _, err := DeclareReferenceScores(ctx, store, model, []ReferenceScore{invalid}); err == nil {
		t.Fatal("empty metric was accepted")
	}
}
