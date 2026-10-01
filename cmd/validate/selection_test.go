package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestCycleRunsWhatPromotionCompares scopes the re-runs a move of every
// surface forces to what promotion compares. In a fixture repository the
// promoted branch holds the commit that the baseline scorecard, published
// beside the store, measured; its one row holds the model's held-out
// accuracy on store/mmlu-pro under the chat template. The model's guard,
// serving cell and training proof run and block promotion, and so does its
// mmlu-pro, the stale compared contract, which takes the chat template the
// baseline compares though its newest record ran raw. Its bbh, never run,
// and its ifeval, stale but compared by no baseline, run in the background,
// each saying why, and a benchmark another model's declaration excludes is
// skipped though its surface moved. Without a baseline every benchmark is
// background. A blocking run leaves the background cells alone, and -run
// -background runs them alone.
func TestCycleRunsWhatPromotionCompares(t *testing.T) {
	t.Parallel()
	root, _, promoted, _ := surfaceHistory(t)
	store := filepath.Join(t.TempDir(), "overgodb-store")
	model, source := testutil.ArtifactID(t, artifact.KindModel, "compared"), testutil.ArtifactID(t, artifact.KindEvidence, "compared")
	chat, raw := evaluation.PromptingChatTemplate.Label(), evaluation.PromptingRawCompletion.Label()
	cells := map[metric.Name]scorecard.Cell{}
	for _, column := range metric.Columns {
		cells[column] = scorecard.Cell{Value: 1, Source: source}
	}
	cells[metric.Quality] = scorecard.Cell{Value: 0.5, Label: "accuracy, store/mmlu-pro (chat-template)", Source: source, Direction: runrecord.DirectionMaximize,
		Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "store/mmlu-pro", Protocol: chat}}
	row := scorecard.Row{Model: model, Name: "compared", Task: recipe.TaskInference, Cells: cells}
	if _, err := scorecard.Publish(scorecard.Directory(store), time.Now(), scorecard.Record{Commit: promoted, Rows: []scorecard.Row{row}}); err != nil {
		t.Fatal(err)
	}
	if output, err := exec.Command("git", "-C", root, "update-ref", scorecard.PromotedBranch, promoted).CombinedOutput(); err != nil {
		t.Fatalf("promote %s: %v: %s", promoted, err, output)
	}
	baseline, err := promotionBaseline(t.Context(), root, store)
	if err != nil || baseline == nil || baseline.Commit != promoted {
		t.Fatalf("the baseline scorecard beside the store was not read: %+v, %v", baseline, err)
	}

	cell := func(id artifact.ID, task recipe.Task, validation string, surface SurfaceID, protocol string) ModelValidation {
		return ModelValidation{
			Model: id, ModelName: "compared", Kind: task, Validation: validation, Surface: surface,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, id.String()+string(task)), Protocol: protocol,
		}
	}
	excluded := cell(testutil.ArtifactID(t, artifact.KindModel, "excluded"), recipe.TaskInference, "mmlu-pro", "evaluation", raw)
	registered := []ModelValidation{
		cell(model, recipe.TaskInference, guardValidation, "inference", ""), cell(model, recipe.TaskInference, servingValidation, "inference", ""),
		cell(model, recipe.TaskInference, "mmlu-pro", "evaluation", raw), cell(model, recipe.TaskInference, "bbh", "evaluation", raw),
		cell(model, recipe.TaskInference, "ifeval", "evaluation", chat), cell(model, recipe.TaskTraining, "training-proof", "training", ""),
		excluded,
	}
	registered = compareContracts(registered, baseline)
	if protocols := []string{registered[2].Protocol, registered[3].Protocol, registered[4].Protocol}; !slices.Equal(protocols, []string{chat, raw, chat}) {
		t.Fatalf("mmlu-pro, bbh and ifeval take protocols %v; want only the compared mmlu-pro to take the baseline's %s", protocols, chat)
	}
	read := acceptedRecords{bound: map[boundKey]error{
		{registered[3].Recipe, "bbh"}:          errors.New("no bbh record for this recipe yet"),
		{excluded.Recipe, excluded.Validation}: fmt.Errorf("%w: it declares dna", errCellInapplicable),
	}}
	moved := AffectRecord{Affected: true, Reason: "authority path internal/cuda/driver.go moves every surface"}
	selected := func(baseline *scorecard.Record) Plan {
		return SelectValidation(Inputs{
			AffectedSurfaces: map[SurfaceID]AffectRecord{"inference": moved, "evaluation": moved, "training": moved},
			Registered:       registered, Current: read.current, Scorecard: baseline,
		})
	}
	scheduled := func(bucket []Selected, scope string) []string {
		var validations []string
		for _, sel := range bucket {
			if !strings.HasPrefix(sel.Promotion, scope+": ") {
				t.Errorf("%s is scheduled %s but says %q", sel.Validation, scope, sel.Promotion)
			}
			validations = append(validations, sel.Validation)
		}
		slices.Sort(validations)
		return validations
	}

	plan := selected(baseline)
	if got, want := scheduled(plan.Run, "blocking"), []string{guardValidation, "mmlu-pro", servingValidation, "training-proof"}; !slices.Equal(got, want) {
		t.Fatalf("promotion waits on %v, want %v: run=%+v background=%+v", got, want, plan.Run, plan.Background)
	}
	if got, want := scheduled(plan.Background, "background"), []string{"bbh", "ifeval"}; !slices.Equal(got, want) {
		t.Fatalf("background cells %v, want %v: run=%+v background=%+v", got, want, plan.Run, plan.Background)
	}
	for _, sel := range plan.Background {
		if !strings.Contains(sel.Promotion, "compares accuracy on store/mmlu-pro (chat-template), not "+sel.Validation) {
			t.Fatalf("background %s does not say what the baseline compares instead: %q", sel.Validation, sel.Promotion)
		}
	}
	if plan.Scorecard != baseline.Name || len(plan.SkipInappropriate) != 1 || plan.SkipInappropriate[0].Model != excluded.Model {
		t.Fatalf("scorecard %q (want %q), skipped %+v: the excluded benchmark was not skipped", plan.Scorecard, baseline.Name, plan.SkipInappropriate)
	}
	if unscoped := selected(nil); !slices.Equal(scheduled(unscoped.Background, "background"), []string{"bbh", "ifeval", "mmlu-pro"}) {
		t.Fatalf("without a baseline a benchmark blocked promotion: run=%+v background=%+v", unscoped.Run, unscoped.Background)
	}

	var out bytes.Buffer
	for background, want := range map[bool][]string{false: scheduled(plan.Run, "blocking"), true: scheduled(plan.Background, "background")} {
		// No cell has a location, so each is reported unbound and none starts.
		err := executeRun(t.Context(), &out, root, store, plan, background)
		for _, validation := range []string{guardValidation, servingValidation, "mmlu-pro", "training-proof", "bbh", "ifeval"} {
			if named := err != nil && strings.Contains(err.Error(), "compared "+validation+":"); named != slices.Contains(want, validation) {
				t.Fatalf("-background=%v took %s: %v, want only %v", background, validation, err, want)
			}
		}
	}
}

// TestEveryComparedContractHasACell holds promotion to a fresh measurement of
// every held-out contract the baseline scorecard compares, whatever family
// its suite is. The baseline compares a text model on store/mmlu under the
// chat template, a family no inference activation is benchmarked on by
// default; a DNA model on store/dna; another text model on store/mmlu-pro;
// and a projection on a set outside the derived catalog. After a move of
// every surface, each compared contract is one blocking cell under the
// protocol compared, re-run by one evaluate worker on its family; the DNA
// model's declaration skips its text suites but never dna; the other
// benchmarks run in the background; and the projection gains no benchmark.
func TestEveryComparedContractHasACell(t *testing.T) {
	t.Parallel()
	root, _, _, _ := surfaceHistory(t)
	const store = "C:/stores/selected"
	chat, raw := evaluation.PromptingChatTemplate.Label(), evaluation.PromptingRawCompletion.Label()
	var suites []evaluation.CompiledSuite
	for _, family := range []string{"mmlu", "mmlu-pro", "bbh", "ifeval", "dna"} {
		data, err := json.Marshal(evaluation.ExactSuite{Schema: "fixture/exact", Source: "store/" + family, Cases: []evaluation.ExactCase{
			{Name: "one", Prompt: "fixture input", MaxTokens: 2, Text: "ok", PromptTokens: 2, GeneratedTokens: 2},
		}})
		if err != nil {
			t.Fatal(err)
		}
		suite, err := evaluation.CompileSuite(data, evaluation.ListingAuthorities())
		if err != nil {
			t.Fatal(err)
		}
		suites = append(suites, suite)
	}

	source := testutil.ArtifactID(t, artifact.KindEvidence, "baseline")
	baseline := &scorecard.Record{Name: "baseline"}
	var registered []ModelValidation
	for _, row := range []struct {
		name, modality, suite, protocol string
		task                            recipe.Task
	}{
		{"Qwen3.8-27B", "", "store/mmlu", chat, recipe.TaskInference},
		{"Carbon-500M", "dna", "store/dna", raw, recipe.TaskInference},
		{"Qwen3.5-4B", "", "store/mmlu-pro", chat, recipe.TaskInference},
		{"gemma-4-E4B-it", "", testutil.ArtifactID(t, artifact.KindDataset, "captions").String(), raw, recipe.TaskProjection},
	} {
		model, activation := testutil.ArtifactID(t, artifact.KindModel, row.name), testutil.ArtifactID(t, artifact.KindRecipe, row.name+"/"+string(row.task))
		baseline.Rows = append(baseline.Rows, scorecard.Row{Model: model, Name: row.name, Task: row.task, Recipe: activation, Cells: map[metric.Name]scorecard.Cell{
			metric.Quality: {Value: 0.5, Source: source, Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: row.suite, Protocol: row.protocol}},
		}})
		for _, validation := range kindValidation(row.task) {
			registered = append(registered, ModelValidation{
				Model: model, ModelName: row.name, Location: "C:/models/" + row.name + ".gguf", Kind: row.task, Modality: row.modality,
				Recipe: activation, Validation: validation.Validation, Surface: validation.Surface,
			})
		}
	}
	registered = compareContracts(registered, baseline)

	// Each benchmark is judged by its family's suite in the catalog and the
	// model's declared domains; no record measures any of them yet.
	read := acceptedRecords{bound: map[boundKey]error{}, directories: map[SurfaceID][]string{}, benchmarks: &benchmarkEvidence{suites: suites}}
	for index, cell := range registered {
		if cell.Kind != recipe.TaskInference || cell.Surface != "evaluation" {
			continue
		}
		judged, err := read.judgeBound(t.Context(), root, nil, cell)
		if err != nil {
			t.Fatal(err)
		}
		registered[index] = judged
	}
	moved := AffectRecord{Affected: true, Reason: "authority path internal/cuda/driver.go moves every surface"}
	plan := SelectValidation(Inputs{
		AffectedSurfaces: map[SurfaceID]AffectRecord{"inference": moved, "evaluation": moved},
		Registered:       registered, Current: read.current, Scorecard: baseline,
	})

	benchmarks := func(bucket []Selected) []string {
		var cells []string
		for _, sel := range bucket {
			if sel.Kind == recipe.TaskInference && sel.Surface == "evaluation" {
				cells = append(cells, sel.ModelName+" "+sel.Validation)
			}
		}
		slices.Sort(cells)
		return cells
	}
	if got, want := benchmarks(plan.Run), []string{"Carbon-500M dna", "Qwen3.5-4B mmlu-pro", "Qwen3.8-27B mmlu"}; !slices.Equal(got, want) {
		t.Fatalf("promotion waits on benchmarks %v, want the compared %v", got, want)
	}
	if got, want := benchmarks(plan.SkipInappropriate), []string{"Carbon-500M bbh", "Carbon-500M ifeval", "Carbon-500M mmlu-pro"}; !slices.Equal(got, want) {
		t.Fatalf("the DNA model's declaration skipped %v, want its text suites %v", got, want)
	}
	if got, want := benchmarks(plan.Background), []string{"Qwen3.5-4B bbh", "Qwen3.5-4B ifeval", "Qwen3.8-27B bbh", "Qwen3.8-27B ifeval", "Qwen3.8-27B mmlu-pro"}; !slices.Equal(got, want) {
		t.Fatalf("background benchmarks %v, want %v", got, want)
	}
	if projection := slices.IndexFunc(registered, func(cell ModelValidation) bool { return cell.Kind == recipe.TaskProjection }); len(registered) != 3*len(kindValidation(recipe.TaskInference))+2+1 ||
		registered[projection].Validation != "projection-proof" {
		t.Fatalf("the compared contracts added cells other than mmlu and dna: %+v", registered)
	}

	for _, row := range baseline.Rows[:3] {
		contract := row.Cells[metric.Quality].Contract
		at := slices.IndexFunc(plan.Run, func(sel Selected) bool {
			return sel.Model == row.Model && sel.Surface == "evaluation" && "store/"+sel.Validation == contract.Suite
		})
		if at < 0 {
			t.Fatalf("%s's compared %s has no blocking cell: run=%+v", row.Name, contract.Suite, plan.Run)
		}
		sel := plan.Run[at]
		if sel.Protocol != contract.Protocol || !strings.HasPrefix(sel.Promotion, "blocking: the baseline scorecard compares accuracy on "+contract.Suite+" ("+contract.Protocol+")") {
			t.Fatalf("%s's %s cell runs under %s and says %q, want the compared %s", row.Name, sel.Validation, sel.Protocol, sel.Promotion, contract.Protocol)
		}
		name, args, err := reacquireCommand(sel.ModelValidation, store)
		want := []string{
			"run", "./cmd/evaluate", "-all", "-worker", "-repo", store, "-model-path", sel.Location,
			"-family", sel.Validation, "-budget", benchmarkBudget.String(),
		}
		if contract.Protocol == chat {
			want = append(want, "-chat-protocol")
		}
		if err != nil || name != "go" || !slices.Equal(args, want) {
			t.Fatalf("%s's %s re-runs as (%q, %v, %v), want go %v", row.Name, sel.Validation, name, args, err, want)
		}
	}
}

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
