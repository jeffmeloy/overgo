package main

import (
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/testutil"
)

// TestCycleLanesRunUntilThePromotedCommitHoldsThem selects the improvement
// cycle's lanes the way it selects cells. In a fixture repository a source
// change under internal/inference, a package of the models lane, is followed
// by a notes-only commit, so that commit's range reaches no lane. Without a
// promoted branch every lane runs. While the promoted branch sits before the
// source change, the models lane runs, since its pass there no longer holds,
// and the lanes nothing reached are reused. Once the branch holds the source
// change every lane is reused, and a range that reaches a lane runs it
// whatever the branch holds.
func TestCycleLanesRunUntilThePromotedCommitHoldsThem(t *testing.T) {
	t.Parallel()
	root, beforeSurface, beforeDocs, head := surfaceHistory(t)
	selected := func(changed []string) map[string]SelectedSuite {
		t.Helper()
		suites, err := cycleSuites(t.Context(), root, head, changed)
		if err != nil {
			t.Fatal(err)
		}
		lanes := map[string]SelectedSuite{}
		for _, suite := range suites {
			lanes[suite.Lane] = suite
		}
		return lanes
	}
	promote := func(commit string) {
		t.Helper()
		if output, err := exec.Command("git", "-C", root, "update-ref", scorecard.PromotedBranch, commit).CombinedOutput(); err != nil {
			t.Fatalf("promote %s: %v: %s", commit, err, output)
		}
	}
	notes, source := []string{"docs/notes.md"}, []string{"internal/inference/inference.go"}
	if exec.Command("git", "-C", root, "rev-parse", "--verify", "--quiet", scorecard.PromotedBranch).Run() != nil {
		for name, lane := range selected(notes) {
			if !lane.Run || !strings.Contains(lane.Reason, "holds its pass yet") {
				t.Errorf("with no promoted commit, lane %s = %+v, want it run", name, lane)
			}
		}
	}
	promote(beforeSurface)
	lanes := selected(notes)
	if models := lanes["models"]; !models.Run || !strings.Contains(models.Reason, "changes since the promoted commit") {
		t.Errorf("a source change since the promoted commit left the models lane %+v", models)
	}
	if audio := lanes["audio"]; audio.Run {
		t.Errorf("no change reaches the audio lane, yet it runs: %+v", audio)
	}
	promote(beforeDocs)
	for name, lane := range selected(notes) {
		if lane.Run {
			t.Errorf("the promoted commit holds every change, yet lane %s runs: %+v", name, lane)
		}
	}
	if models := selected(source)["models"]; !models.Run || !strings.Contains(models.Reason, "commit range's changes reach it") {
		t.Errorf("a range reaching the models lane left it %+v", models)
	}
}

// TestSelectValidation covers every decision branch of the pure selection: a
// commit's changes run the cells whose surface moved or whose accepted record
// is not current, reuse only current records, and account for operator-owned,
// registered-inactive and inapplicable cells, with the run bucket ordered
// cheapest first.
func TestSelectValidation(t *testing.T) {
	t.Parallel()
	activation := func(seed string) artifact.ID { return testutil.ArtifactID(t, artifact.KindRecipe, seed) }
	model := func(seed string) artifact.ID { return testutil.ArtifactID(t, artifact.KindModel, seed) }
	read := acceptedRecords{surface: "surface-now", records: map[string]guardRecord{
		locationKey("C:/models/qwen05.gguf"): {Surface: "surface-now", Passed: true},
	}}

	in := Inputs{
		Baseline: "base", Head: "head",
		ChangedPaths: []string{"kernels/cuda/ops_f32.cu"},
		AffectedSurfaces: map[SurfaceID]AffectRecord{
			"image":     {Affected: true, Reason: "kernels/cuda/ops_f32.cu changed"},
			"inference": {Affected: false, Reason: ""},
		},
		Registered: []ModelValidation{
			// Affected image cell with a big cost -> Run, but ordered after the cheap one.
			{Model: model("sensenova"), Kind: recipe.Task("image-gen"), Modality: "image", Validation: "media-proof", Surface: "image", Recipe: activation("sn"), Cost: 1200},
			// Affected image cell, cheaper -> Run first.
			{Model: model("un0"), Kind: recipe.Task("image-gen"), Modality: "image", Validation: "media-proof", Surface: "image", Recipe: activation("un0"), Cost: 30},
			// Unaffected inference cell with a current record -> Reuse.
			{Model: model("qwen05"), Location: "C:/models/qwen05.gguf", Kind: recipe.Task("inference"), Modality: "text", Validation: "guard", Surface: "inference", Recipe: activation("q05"), Cost: 10},
			// Unaffected inference cell without a record -> Run (first acquisition).
			{Model: model("newtext"), Location: "C:/models/newtext.gguf", Kind: recipe.Task("inference"), Modality: "text", Validation: "guard", Surface: "inference", Recipe: activation("new"), Cost: 5},
			// Operator-only cell -> OperatorOwned even though its surface is affected.
			{Model: model("qwen27"), Kind: recipe.Task("inference"), Modality: "text", Validation: "mmlu-pro-full", Surface: "inference", OperatorOnly: true, OperatorReason: "full 27B MMLU-Pro is operator-launched"},
			// Registered without activation -> RegisteredInactive.
			{Model: model("carbon"), ModelName: "Carbon-500M"},
			// Activation with no applicable validation -> SkipInappropriate.
			{Model: model("livetrain"), Kind: recipe.Task("training"), Modality: "text"},
		},
		Current: read.current,
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
	t.Parallel()
	plan := SelectValidation(Inputs{Baseline: "b", Head: "h"})
	if plan.Run == nil || plan.Reuse == nil || len(plan.Run) != 0 {
		t.Fatalf("empty plan not clean: %+v", plan)
	}
}

// TestReuseNeedsCurrentEvidence holds reuse to evidence that is current at
// this tree, not to a commit range that left a surface alone. A source change
// moved the inference surface and a plan-only commit followed it, so the
// default baseline..HEAD moves no surface. A guard cell whose newest record
// measured the surface before the source change still runs; one whose record
// measured the surface the code is at now is reused. Both carry an active
// recipe, which credits neither. The inventory takes the same decision: the
// first model is activated and names its guard, the second is validated.
func TestReuseNeedsCurrentEvidence(t *testing.T) {
	t.Parallel()
	active := testutil.ArtifactID(t, artifact.KindRecipe, "active-inference")
	older, current := testutil.ArtifactID(t, artifact.KindModel, "older-record"), testutil.ArtifactID(t, artifact.KindModel, "current-record")
	cells := []ModelValidation{
		{Model: older, Location: "C:/models/Older/m.gguf", Kind: recipe.TaskInference, Validation: guardValidation, Surface: "inference", Recipe: active},
		{Model: current, Location: "C:/models/Current/m.gguf", Kind: recipe.TaskInference, Validation: guardValidation, Surface: "inference", Recipe: active},
	}
	const beforeSource, afterSource = "a0a0a0a0a0a0a0a0-before-the-source-change", "b1b1b1b1b1b1b1b1-after-the-source-change"
	read := acceptedRecords{surface: afterSource, records: map[string]guardRecord{
		locationKey(cells[0].Location): {Location: locationKey(cells[0].Location), Surface: beforeSource, Commit: "source-change", Passed: true},
		locationKey(cells[1].Location): {Location: locationKey(cells[1].Location), Surface: afterSource, Commit: "plan-only", Passed: true},
	}}
	plan := SelectValidation(Inputs{
		Baseline: "source-change", Head: "plan-only", ChangedPaths: []string{"docs/plan.json"},
		AffectedSurfaces: map[SurfaceID]AffectRecord{"inference": {}},
		Registered:       cells, Current: read.current,
	})
	if len(plan.AffectedSurfaces) != 0 {
		t.Fatalf("a plan-only commit moved %v", plan.AffectedSurfaces)
	}
	if len(plan.Run) != 1 || plan.Run[0].Model != older || !strings.Contains(plan.Run[0].Reason, "measured inference surface "+beforeSource[:surfaceShown]) {
		t.Fatalf("a record bound to an older surface was not selected to run: run=%+v reuse=%+v", plan.Run, plan.Reuse)
	}
	if len(plan.Reuse) != 1 || plan.Reuse[0].Model != current {
		t.Fatalf("a record bound to the current surface was not reused: reuse=%+v run=%+v", plan.Reuse, plan.Run)
	}

	directories := []modelDirectory{{Path: "C:/models/Older", Weights: true}, {Path: "C:/models/Current", Weights: true}}
	inventory := buildInventory(directories, cells, nil, read.current, "C:/stores/selected")
	// Sorted by directory: Current, then Older.
	if inventory[0].Standing != standingValidated || inventory[1].Standing != standingActivated || len(inventory[1].Commands) != 1 {
		t.Fatalf("the inventory disagrees with the plan: %+v", inventory)
	}
}
