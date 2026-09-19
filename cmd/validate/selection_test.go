package main

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestSelectValidation covers every decision branch of the pure selection: a
// commit's changes run only the cells whose surface moved (or that lack
// evidence), reuse the rest, and account for operator-owned, registered-inactive
// and inapplicable cells, with the run bucket ordered cheapest first.
func TestSelectValidation(t *testing.T) {
	evidence := func(seed string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvidence, seed) }
	model := func(seed string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, seed) }

	in := Inputs{
		Baseline: "base", Head: "head",
		ChangedPaths: []string{"kernels/cuda/ops_f32.cu"},
		AffectedSurfaces: map[SurfaceID]AffectRecord{
			"image":     {Affected: true, Reason: "kernels/cuda/ops_f32.cu changed"},
			"inference": {Affected: false, Reason: ""},
		},
		Registered: []ModelValidation{
			// Affected image cell with a big cost -> Run, but ordered after the cheap one.
			{Model: model("sensenova"), Kind: recipe.Task("image-gen"), Modality: "image", Validation: "media-proof", Surface: "image", Evidence: evidence("sn"), Cost: 1200},
			// Affected image cell, cheaper -> Run first.
			{Model: model("un0"), Kind: recipe.Task("image-gen"), Modality: "image", Validation: "media-proof", Surface: "image", Evidence: evidence("un0"), Cost: 30},
			// Unaffected inference cell with evidence -> Reuse.
			{Model: model("qwen05"), Kind: recipe.Task("inference"), Modality: "text", Validation: "guard", Surface: "inference", Evidence: evidence("q05"), Cost: 10},
			// Unaffected inference cell without evidence -> Run (first acquisition).
			{Model: model("newtext"), Kind: recipe.Task("inference"), Modality: "text", Validation: "guard", Surface: "inference", Cost: 5},
			// Operator-only cell -> OperatorOwned even though its surface is affected.
			{Model: model("qwen27"), Kind: recipe.Task("inference"), Modality: "text", Validation: "mmlu-pro-full", Surface: "inference", OperatorOnly: true, OperatorReason: "full 27B MMLU-Pro is operator-launched"},
			// Registered without activation -> RegisteredInactive.
			{Model: model("carbon"), ModelName: "Carbon-500M"},
			// Activation with no applicable validation -> SkipInappropriate.
			{Model: model("livetrain"), Kind: recipe.Task("training"), Modality: "text"},
		},
	}

	plan := SelectValidation(in)

	if got := len(plan.Run); got != 3 {
		t.Fatalf("run=%d want 3: %+v", got, plan.Run)
	}
	// Cheapest first: newtext (5, no evidence) < un0 (30) < sensenova (1200).
	if plan.Run[0].ModelName == "" && plan.Run[0].Model != in.Registered[3].Model {
		t.Fatalf("run not ordered cheapest first: %s", plan.Run[0].Model)
	}
	if plan.Run[0].Cost > plan.Run[1].Cost || plan.Run[1].Cost > plan.Run[2].Cost {
		t.Fatalf("run costs not ascending: %d %d %d", plan.Run[0].Cost, plan.Run[1].Cost, plan.Run[2].Cost)
	}
	if len(plan.Reuse) != 1 || plan.Reuse[0].Validation != "guard" {
		t.Fatalf("reuse=%+v", plan.Reuse)
	}
	if len(plan.OperatorOwned) != 1 || !plan.OperatorOwned[0].OperatorOnly {
		t.Fatalf("operator-owned=%+v", plan.OperatorOwned)
	}
	if len(plan.RegisteredInactive) != 1 || plan.RegisteredInactive[0].ModelName != "Carbon-500M" {
		t.Fatalf("registered-inactive=%+v", plan.RegisteredInactive)
	}
	if len(plan.SkipInappropriate) != 1 {
		t.Fatalf("skip-inappropriate=%+v", plan.SkipInappropriate)
	}
	if len(plan.AffectedSurfaces) != 1 || plan.AffectedSurfaces[0] != "image" {
		t.Fatalf("affected surfaces=%v", plan.AffectedSurfaces)
	}
}

// TestSelectValidationStableEmpty proves an empty registration set yields empty,
// non-nil buckets (a clean commit with nothing to run reads as covered).
func TestSelectValidationStableEmpty(t *testing.T) {
	plan := SelectValidation(Inputs{Baseline: "b", Head: "h"})
	if plan.Run == nil || plan.Reuse == nil || len(plan.Run) != 0 {
		t.Fatalf("empty plan not clean: %+v", plan)
	}
}
