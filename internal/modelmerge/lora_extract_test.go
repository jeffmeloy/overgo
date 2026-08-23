package modelmerge

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/tensor"
	"overgo/internal/testutil"
)

func TestLoRAExtractionExactBase(t *testing.T) {
	base, tuned := loraSnapshots(t)
	policy := loraPolicy()
	if _, err := ExtractLoRA(base, tuned, policy); err != nil {
		t.Fatal(err)
	}
	wrong := tuned
	wrong.Base = testutil.ArtifactID(t, artifact.KindModel, "wrong LoRA base")
	wrong.ID, _ = (Compiler{}).identity(wrong)
	if _, err := ExtractLoRA(base, wrong, policy); err == nil {
		t.Fatal("LoRA extraction accepted the wrong exact base")
	}
}

func TestLoRAExtractionResidualBudget(t *testing.T) {
	base, tuned := loraSnapshots(t)
	extraction, err := ExtractLoRA(base, tuned, loraPolicy())
	if err != nil {
		t.Fatal(err)
	}
	factors := extraction.Weights["weight"]
	if factors.Rank != uint32(tensor.SingletonExtent) || factors.RelativeResidual > extraction.Policy.MaximumRelativeResidual {
		t.Fatalf("derived factors = %+v", factors)
	}
}

func TestLoRAExtractionFrozenReconstruction(t *testing.T) {
	base, tuned := loraSnapshots(t)
	extraction, err := ExtractLoRA(base, tuned, loraPolicy())
	if err != nil {
		t.Fatal(err)
	}
	reconstructed, err := extraction.Reconstruct(base)
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range reconstructed.Tensors["weight"].Values {
		if math.Abs(float64(value-tuned.Tensors["weight"].Values[index])) > extraction.Policy.MaximumRelativeResidual {
			t.Fatalf("reconstructed[%d]=%g tuned=%g", index, value, tuned.Tensors["weight"].Values[index])
		}
	}
	if base.Tensors["weight"].Values[0] != 1 {
		t.Fatal("frozen base mutated during reconstruction")
	}
}

func TestLoRAExtractionLineage(t *testing.T) {
	base, tuned := loraSnapshots(t)
	extraction, err := ExtractLoRA(base, tuned, loraPolicy())
	if err != nil {
		t.Fatal(err)
	}
	if extraction.ID.Kind() != artifact.KindAdapter {
		t.Fatalf("adapter identity = %s", extraction.ID)
	}
	parents := map[artifact.ID]bool{}
	for _, edge := range extraction.Lineage() {
		parents[edge.Parent] = true
	}
	if !parents[base.ID] || !parents[tuned.ID] || !parents[base.Definition] || len(parents) != tensor.TripleExtent {
		t.Fatalf("LoRA lineage = %+v", extraction.Lineage())
	}
}

func loraSnapshots(t *testing.T) (Snapshot, Snapshot) {
	t.Helper()
	compiler := Compiler{}
	definition := testutil.ArtifactID(t, artifact.KindModelDefinition, "LoRA extraction definition")
	base, err := compiler.Seal(definition, artifact.ID{}, map[string]Weight{
		"weight": {Layout: tensor.MustShape(3, 2), Values: []float32{1, 2, 3, 4, 5, 6}},
	})
	if err != nil {
		t.Fatal(err)
	}
	tuned, err := compiler.Seal(definition, base.ID, map[string]Weight{
		"weight": {Layout: tensor.MustShape(3, 2), Values: []float32{3, 4, 5, 8, 9, 10}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return base, tuned
}

func loraPolicy() LoRAExtractionPolicy {
	return LoRAExtractionPolicy{
		MaximumRelativeResidual: 1e-5,
		Rationale:               "retain the minimum rank meeting measured frozen-base reconstruction error",
		ReopenTrigger:           "re-extract when the exact base, tuned weights, or reconstruction budget changes",
	}
}
