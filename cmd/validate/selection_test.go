package main

import (
	"bytes"
	"encoding/json"
	"maps"
	"os"
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

// fixtureRepository writes a repository of one commit, which a test promotes,
// and returns its root with that commit.
func fixtureRepository(t *testing.T) (root, head string) {
	t.Helper()
	root = t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "notes.md"), []byte("fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, arguments := range [][]string{{"init", "-q"}, {"add", "-A"}, {"commit", "-q", "-m", "fixture"}, {"rev-parse", "HEAD"}} {
		command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...)
		command.Dir = root
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
		head = strings.TrimSpace(string(output))
	}
	return root, head
}

// TestCycleRunsWhatPromotionCompares scopes the cells that run, every
// activated cell, to what promotion compares. In a fixture repository the
// promoted branch holds the commit that the baseline scorecard, published
// beside the store, measured; its one row holds the model's held-out
// accuracy on store/mmlu-pro under the chat template. The model's guard,
// serving cell and training proof run and block promotion, and so does its
// mmlu-pro, the compared contract, which takes the chat template the
// baseline compares though its newest record ran raw. Its bbh and its
// ifeval, compared by no baseline, run in the background,
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
	root, promoted := fixtureRepository(t)
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

	cell := func(id artifact.ID, task recipe.Task, validation, protocol string) ModelValidation {
		return ModelValidation{
			Model: id, ModelName: "compared", Kind: task, Validation: validation,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, id.String()+string(task)), Protocol: protocol,
		}
	}
	excluded := cell(testutil.ArtifactID(t, artifact.KindModel, "excluded"), recipe.TaskInference, "mmlu-pro", raw)
	excluded.Inapplicable = "it declares dna"
	registered := []ModelValidation{
		cell(model, recipe.TaskInference, guardValidation, ""), cell(model, recipe.TaskInference, servingValidation, ""),
		cell(model, recipe.TaskInference, "mmlu-pro", raw), cell(model, recipe.TaskInference, "bbh", raw),
		cell(model, recipe.TaskInference, "ifeval", chat), cell(model, recipe.TaskTraining, "training-proof", ""),
		excluded,
	}
	if compared := compareContracts(slices.Clone(registered), baseline); !slices.Equal([]string{compared[2].Protocol, compared[3].Protocol, compared[4].Protocol}, []string{chat, raw, chat}) {
		t.Fatalf("mmlu-pro, bbh and ifeval take protocols %+v; want only the compared mmlu-pro to take the baseline's %s", compared[2:5], chat)
	}
	selected := func(baseline *scorecard.Record) Plan {
		return SelectValidation(Inputs{Registered: compareContracts(slices.Clone(registered), baseline), Scorecard: baseline})
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
	for scope, want := range map[runScope][]string{scopeCycle: names(plan.Run), scopeCells: names(plan.Run), scopeBackground: names(plan.Background), scopeLanes: nil} {
		// No cell has a location, so each is reported unbound and none starts.
		err := executeRun(t.Context(), &out, root, store, nil, plan, scope)
		for _, validation := range []string{guardValidation, servingValidation, "mmlu-pro", "training-proof", "bbh", "ifeval"} {
			if named := err != nil && strings.Contains(err.Error(), "compared "+validation+":"); named != slices.Contains(want, validation) {
				t.Fatalf("scope %d took %s: %v, want only %v", scope, validation, err, want)
			}
		}
	}
	for _, flags := range []struct{ background, cells, lanes bool }{{true, true, false}, {true, false, true}, {false, true, true}} {
		if _, err := runScopeOf(flags.background, flags.cells, flags.lanes); err == nil {
			t.Errorf("flags %+v select one run, want them refused", flags)
		}
	}
}

// TestEveryComparedContractHasACell holds promotion to a fresh measurement of
// every held-out contract the baseline scorecard compares, whatever family
// its suite is. The baseline compares a text model on store/mmlu under the
// chat template, a family no inference activation is benchmarked on by
// default; a DNA model on store/dna; another text model on store/mmlu-pro;
// and a projection on a set outside the derived catalog. Each compared
// contract is one blocking cell under the
// protocol compared, re-run by one evaluate worker on its family; the DNA
// model's declaration skips its text suites but never dna; the other
// benchmarks run in the background; and the projection gains no benchmark.
func TestEveryComparedContractHasACell(t *testing.T) {
	t.Parallel()
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
				Recipe: activation, Validation: validation,
			})
		}
	}
	registered = compareContracts(registered, baseline)

	// Each benchmark is bound by its family's suite in the catalog and the
	// model's declared domains; no record measures any of them yet.
	catalog := benchmarkEvidence{suites: suites}
	for index, cell := range registered {
		if !benchmark(cell) {
			continue
		}
		bound, err := catalog.bind(t.Context(), nil, cell)
		if err != nil {
			t.Fatal(err)
		}
		registered[index] = bound
	}
	plan := SelectValidation(Inputs{Registered: registered, Scorecard: baseline})

	benchmarks := func(bucket []Selected) []string {
		var cells []string
		for _, sel := range bucket {
			if benchmark(sel.ModelValidation) {
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
			return sel.Model == row.Model && benchmark(sel.ModelValidation) && "store/"+sel.Validation == contract.Suite
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

// TestSelectValidation covers every decision branch of the pure selection:
// every activated cell its model's declaration does not exclude runs, every
// cycle, whatever it last recorded. A guard whose record measured today's
// digest, a serving cell with a measured attempt, a DNA model's smoke run and
// a benchmark whose record succeeded run as one that recorded nothing does,
// and a benchmark that has a record names -replay of it, whatever made it
// run; operator-owned, registered-inactive and inapplicable cells are
// accounted for; and every bucket is ordered cheapest first.
func TestSelectValidation(t *testing.T) {
	t.Parallel()
	const store = "C:/stores/selected"
	previous := testutil.ArtifactID(t, artifact.KindEvaluation, "previous")
	cell := func(name, validation string, cost int64) ModelValidation {
		return ModelValidation{
			Model: testutil.ArtifactID(t, artifact.KindModel, name), ModelName: name, Location: "C:/models/" + name + ".gguf",
			Kind: recipe.TaskInference, Modality: "text", Validation: validation,
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name), Cost: cost,
		}
	}
	digest, unmeasured, served := cell("digest", guardValidation, 10), cell("unmeasured", guardValidation, 5), cell("served", servingValidation, 20)
	dna := cell("dna", guardValidation, 3)
	dna.Modality = dnaModality
	succeeded, stale, unrecorded := cell("succeeded", "mmlu-pro", 1200), cell("stale", "mmlu-pro", 30), cell("unrecorded", "mmlu-pro", 40)
	succeeded.Replay, stale.Replay, stale.Stale = previous, previous, "its recipe was replaced"
	excluded := cell("excluded", "mmlu-pro", 1)
	excluded.Inapplicable = "it declares dna"
	operated := cell("operated", "mmlu-pro-full", 0)
	operated.OperatorOnly, operated.OperatorReason = true, "full 27B MMLU-Pro is operator-launched"
	plan := SelectValidation(Inputs{Registered: []ModelValidation{
		succeeded, digest, unmeasured, served, dna, stale, unrecorded, operated, excluded,
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
		{"run", names(plan.Run), "dna unmeasured digest served"},
		{"background", names(plan.Background), "stale unrecorded succeeded"},
		{"operator-owned", names(plan.OperatorOwned), "operated"},
		{"registered-inactive", names(plan.RegisteredInactive), "Carbon-500M"},
		{"skip-inappropriate", names(plan.SkipInappropriate), "livetrain excluded"},
	} {
		if bucket.got != bucket.want {
			t.Errorf("%s holds %q, want %q", bucket.name, bucket.got, bucket.want)
		}
	}
	for _, sel := range plan.Run {
		if sel.Reason != "run: re-acquired every cycle" {
			t.Errorf("%s runs for %q, want every cycle", sel.ModelName, sel.Reason)
		}
	}
	if !strings.Contains(plan.Background[0].Reason, "the activation is stale") {
		t.Errorf("the stale benchmark runs for %q, want its stale activation said", plan.Background[0].Reason)
	}
	for _, sel := range slices.Concat(plan.Run, plan.Background) {
		_, args, err := reacquireCommand(sel.ModelValidation, store)
		at := slices.Index(args, "-replay")
		if err != nil || (at >= 0) != sel.Replay.Valid() || at >= 0 && args[at+1] != previous.String() {
			t.Errorf("%s re-runs as %v, %v, want -replay %s exactly when it has a record", sel.ModelName, args, err, previous)
		}
	}
}

// TestSelectValidationStableEmpty proves an empty registration set yields empty,
// non-nil buckets (a clean commit with nothing to run reads as covered).
func TestSelectValidationStableEmpty(t *testing.T) {
	t.Parallel()
	plan := SelectValidation(Inputs{})
	if plan.Run == nil || plan.Background == nil || plan.SkipInappropriate == nil || len(plan.Run) != 0 {
		t.Fatalf("empty plan not clean: %+v", plan)
	}
}
