package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
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

// TestCycleRunsWhatPromotionCompares scopes the cells that run, none of them
// current at this tree, to what promotion compares. In a fixture repository the
// promoted branch holds the commit that the baseline scorecard, published
// beside the store, measured; its one row holds the model's held-out
// accuracy on store/mmlu-pro under the chat template. The model's guard,
// serving cell and training proof run and block promotion, and so does its
// mmlu-pro, the stale compared contract, which takes the chat template the
// baseline compares though its newest record ran raw. Its bbh, never run,
// and its ifeval, stale but compared by no baseline, run in the background,
// each saying why, and a benchmark another model's declaration excludes is
// skipped. A baseline that retains another held-out
// contract beside its headline, raw completion on mmlu-pro or the chat
// template on ifeval, blocks promotion on that contract too, whichever of
// them is the headline, while a benchmark no contract names stays in the
// background. Without a baseline every benchmark is background. A blocking
// run leaves the background cells alone, and -run -background runs them
// alone.
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
	if compared := compareContracts(slices.Clone(registered), baseline); !slices.Equal([]string{compared[2].Protocol, compared[3].Protocol, compared[4].Protocol}, []string{chat, raw, chat}) {
		t.Fatalf("mmlu-pro, bbh and ifeval take protocols %+v; want only the compared mmlu-pro to take the baseline's %s", compared[2:5], chat)
	}
	read := acceptedRecords{bound: map[boundKey]error{
		{registered[3].Recipe, "bbh", raw}:                        errors.New("no bbh record for this recipe yet"),
		{excluded.Recipe, excluded.Validation, excluded.Protocol}: fmt.Errorf("%w: it declares dna", errCellInapplicable),
	}}
	selected := func(baseline *scorecard.Record) Plan {
		return SelectValidation(Inputs{Registered: compareContracts(slices.Clone(registered), baseline), Current: read.current, Scorecard: baseline})
	}
	// scheduled names each cell of a bucket by its validation and, for a
	// benchmark, the protocol it runs under.
	scheduled := func(bucket []Selected, scope string) []string {
		var validations []string
		for _, sel := range bucket {
			if !strings.HasPrefix(sel.Promotion, scope+": ") {
				t.Errorf("%s is scheduled %s but says %q", sel.Validation, scope, sel.Promotion)
			}
			validations = append(validations, strings.TrimSpace(sel.Validation+" "+sel.Protocol))
		}
		slices.Sort(validations)
		return validations
	}

	// The baseline holds the model to every held-out contract it measured,
	// the headline and each retained beside it, whichever is the headline:
	// each compared family runs and blocks under each compared protocol, and
	// a benchmark no baseline contract names runs in the background.
	retaining := func(headline scorecard.Contract, others ...scorecard.Contract) *scorecard.Record {
		retained := *baseline
		retained.Rows = slices.Clone(baseline.Rows)
		retained.Rows[0].Cells = maps.Clone(retained.Rows[0].Cells)
		quality := func(contract scorecard.Contract) scorecard.Cell {
			cell := retained.Rows[0].Cells[metric.Quality]
			cell.Contract = contract
			return cell
		}
		retained.Rows[0].Cells[metric.Quality] = quality(headline)
		for _, other := range others {
			retained.Rows[0].Heldout = append(retained.Rows[0].Heldout, quality(other))
		}
		return &retained
	}
	contract := func(suite, protocol string) scorecard.Contract {
		return scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: suite, Protocol: protocol}
	}
	for _, scenario := range []struct {
		name                 string
		baseline             *scorecard.Record
		blocking, background []string
	}{
		{name: "the headline compared", baseline: baseline,
			blocking: []string{guardValidation, "mmlu-pro " + chat, servingValidation, "training-proof"}, background: []string{"bbh " + raw, "ifeval " + chat}},
		{name: "a retained protocol compared beside the headline", baseline: retaining(contract("store/mmlu-pro", chat), contract("store/mmlu-pro", raw)),
			blocking: []string{guardValidation, "mmlu-pro " + chat, "mmlu-pro " + raw, servingValidation, "training-proof"}, background: []string{"bbh " + raw, "ifeval " + chat}},
		{name: "the headline and a retained protocol reordered", baseline: retaining(contract("store/mmlu-pro", raw), contract("store/mmlu-pro", chat)),
			blocking: []string{guardValidation, "mmlu-pro " + chat, "mmlu-pro " + raw, servingValidation, "training-proof"}, background: []string{"bbh " + raw, "ifeval " + chat}},
		{name: "a retained family compared beside the headline", baseline: retaining(contract("store/mmlu-pro", chat), contract("store/ifeval", chat)),
			blocking: []string{guardValidation, "ifeval " + chat, "mmlu-pro " + chat, servingValidation, "training-proof"}, background: []string{"bbh " + raw}},
		{name: "no baseline", baseline: nil,
			blocking: []string{guardValidation, servingValidation, "training-proof"}, background: []string{"bbh " + raw, "ifeval " + chat, "mmlu-pro " + raw}},
	} {
		plan := selected(scenario.baseline)
		blocking, background := scheduled(plan.Run, "blocking"), scheduled(plan.Background, "background")
		if !slices.Equal(blocking, scenario.blocking) || !slices.Equal(background, scenario.background) {
			t.Errorf("%s: promotion waits on %v and not on %v, want %v and not %v", scenario.name, blocking, background, scenario.blocking, scenario.background)
			continue
		}
		for _, sel := range slices.Concat(plan.Run, plan.Background) {
			if !benchmark(sel.ModelValidation) {
				continue
			}
			if measured := "store/" + sel.Validation + " (" + sel.Protocol + ")"; scenario.baseline != nil &&
				!strings.HasPrefix(sel.Promotion, "blocking: the baseline scorecard compares accuracy on "+measured) &&
				!(strings.HasPrefix(sel.Promotion, "background: the baseline scorecard compares accuracy on ") && strings.HasSuffix(sel.Promotion, ", not "+sel.Validation+" ("+sel.Protocol+")")) {
				t.Errorf("%s: %s does not say which contract the baseline compares: %q", scenario.name, sel.Validation, sel.Promotion)
			}
		}
	}

	plan := selected(baseline)
	if plan.Scorecard != baseline.Name || len(plan.SkipInappropriate) != 1 || plan.SkipInappropriate[0].Model != excluded.Model {
		t.Fatalf("scorecard %q (want %q), skipped %+v: the excluded benchmark was not skipped", plan.Scorecard, baseline.Name, plan.SkipInappropriate)
	}
	names := func(bucket []Selected) []string {
		var validations []string
		for _, sel := range bucket {
			validations = append(validations, sel.Validation)
		}
		return validations
	}

	var out bytes.Buffer
	for background, want := range map[bool][]string{false: names(plan.Run), true: names(plan.Background)} {
		// No cell has a location, so each is reported unbound and none starts.
		err := executeRun(t.Context(), &out, root, store, nil, plan, background)
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
// and a projection on a set outside the derived catalog. With no record
// current, each compared contract is one blocking cell under the
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
	plan := SelectValidation(Inputs{Registered: registered, Current: read.current, Scorecard: baseline})

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

// TestSelectValidation covers every decision branch of the pure selection,
// with no commit range among its inputs: a cell runs when its accepted
// record is not current at this tree or its activation is stale, and a
// serving cell and a DNA model's smoke run run every cycle; only a current
// record is reused; a benchmark that runs replays its newest record first,
// whatever made it run; operator-owned, registered-inactive and inapplicable
// cells are accounted for; and every bucket is ordered cheapest first.
func TestSelectValidation(t *testing.T) {
	t.Parallel()
	previous := testutil.ArtifactID(t, artifact.KindEvaluation, "previous")
	cell := func(name, validation string, surface SurfaceID, cost int64) ModelValidation {
		return ModelValidation{
			Model: testutil.ArtifactID(t, artifact.KindModel, name), ModelName: name, Location: "C:/models/" + name + ".gguf",
			Kind: recipe.TaskInference, Modality: "text", Validation: validation, Surface: surface,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name), Cost: cost,
		}
	}
	current, unmeasured, served := cell("current", guardValidation, "inference", 10), cell("unmeasured", guardValidation, "inference", 5), cell("served", servingValidation, "inference", 20)
	dna := cell("dna", guardValidation, "inference", 3)
	dna.Modality = dnaModality
	moved, stale, held := cell("moved", "mmlu-pro", "evaluation", 1200), cell("stale", "mmlu-pro", "evaluation", 30), cell("held", "mmlu-pro", "evaluation", 40)
	moved.Replay, stale.Replay, held.Replay, stale.Stale = previous, previous, previous, "its recipe was replaced"
	operated := cell("operated", "mmlu-pro-full", "evaluation", 0)
	operated.OperatorOnly, operated.OperatorReason = true, "full 27B MMLU-Pro is operator-launched"
	read := acceptedRecords{surface: "surface-now", records: map[string]guardRecord{locationKey(current.Location): {Surface: "surface-now", Passed: true}}, bound: map[boundKey]error{
		{moved.Recipe, moved.Validation, moved.Protocol}: errors.New("its mmlu-pro record ran at commit c0ffee and the evaluation surface moved since"),
		{stale.Recipe, stale.Validation, stale.Protocol}: nil,
		{held.Recipe, held.Validation, held.Protocol}:    nil,
	}}
	plan := SelectValidation(Inputs{Current: read.current, Registered: []ModelValidation{
		moved, current, unmeasured, served, dna, stale, held, operated,
		{Model: testutil.ArtifactID(t, artifact.KindModel, "carbon"), ModelName: "Carbon-500M"},
		{Model: testutil.ArtifactID(t, artifact.KindModel, "livetrain"), ModelName: "livetrain", Kind: recipe.TaskTraining},
	}})

	names := func(bucket []Selected) string {
		var listed []string
		for _, sel := range bucket {
			listed = append(listed, sel.ModelName)
		}
		return strings.Join(listed, " ")
	}
	for _, bucket := range []struct{ name, got, want string }{
		{"run", names(plan.Run), "dna unmeasured served"},
		{"background", names(plan.Background), "stale moved"},
		{"reuse", names(plan.Reuse), "current held"},
		{"operator-owned", names(plan.OperatorOwned), "operated"},
		{"registered-inactive", names(plan.RegisteredInactive), "Carbon-500M"},
		{"skip-inappropriate", names(plan.SkipInappropriate), "livetrain"},
	} {
		if bucket.got != bucket.want {
			t.Errorf("%s holds %q, want %q", bucket.name, bucket.got, bucket.want)
		}
	}
	if every := "run: " + errEveryCycle.Error(); len(plan.Run) != 3 || plan.Run[0].Reason != every || plan.Run[2].Reason != every {
		t.Fatalf("the DNA guard and the serving cell do not run every cycle: %+v", plan.Run)
	}
	for _, sel := range plan.Background {
		if sel.Replay != previous {
			t.Errorf("benchmark %s runs (%s) without replaying its newest record", sel.ModelName, sel.Reason)
		}
	}
}

// TestSelectValidationStableEmpty proves an empty registration set yields empty,
// non-nil buckets (a clean commit with nothing to run reads as covered).
func TestSelectValidationStableEmpty(t *testing.T) {
	t.Parallel()
	plan := SelectValidation(Inputs{})
	if plan.Run == nil || plan.Reuse == nil || len(plan.Run) != 0 {
		t.Fatalf("empty plan not clean: %+v", plan)
	}
}

// TestReuseNeedsCurrentEvidence holds reuse to evidence that is current at
// this tree. A guard cell whose newest record measured the inference surface
// before a source change runs; one whose record measured the surface the
// code is at now is reused. Both carry an active recipe, which credits
// neither. The inventory takes the same decision: the first model is
// activated and names its guard, the second is validated.
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
	plan := SelectValidation(Inputs{Registered: cells, Current: read.current})
	if len(plan.Run) != 1 || plan.Run[0].Model != older || !strings.Contains(plan.Run[0].Reason, "measured inference surface "+beforeSource[:surfaceShown]) {
		t.Fatalf("a record bound to an older surface was not selected to run: run=%+v reuse=%+v", plan.Run, plan.Reuse)
	}
	if len(plan.Reuse) != 1 || plan.Reuse[0].Model != current {
		t.Fatalf("a record bound to the current surface was not reused: reuse=%+v run=%+v", plan.Reuse, plan.Run)
	}

	// The current model's serving cell, re-acquired every cycle whatever it
	// recorded, neither validates nor demotes it.
	serving := cells[1]
	serving.Validation = servingValidation
	directories := []modelDirectory{{Path: "C:/models/Older", Weights: true}, {Path: "C:/models/Current", Weights: true}}
	inventory := buildInventory(directories, append(cells, serving), nil, read.current, "C:/stores/selected")
	// Sorted by directory: Current, then Older.
	if inventory[0].Standing != standingValidated || inventory[1].Standing != standingActivated || len(inventory[1].Commands) != 1 {
		t.Fatalf("the inventory disagrees with the plan: %+v", inventory)
	}
}
