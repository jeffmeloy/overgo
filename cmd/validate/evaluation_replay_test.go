package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// replayFixtureRuntime scores each candidate from its case's prompt alone and
// counts the cases it scores; drift moves one prompt's scores by one unit in
// the last place, a change only an exact float comparison sees.
type replayFixtureRuntime struct {
	scored *int
	drift  string
}

func (replayFixtureRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("the replay fixture scores likelihoods only")
}

func (runtime replayFixtureRuntime) ScoreContinuations(_ context.Context, prompt string, candidates []string) ([]sequencescore.Score, error) {
	*runtime.scored++
	scores := make([]sequencescore.Score, len(candidates))
	for index := range candidates {
		value := -float64(len(prompt)%(index+3)+1) / 3
		if prompt == runtime.drift {
			value = math.Nextafter(value, 0)
		}
		scores[index] = sequencescore.Score{LogProbability: value, Tokens: 1}
	}
	return scores, nil
}

// replayFixtureSuite is a raw-completion multiple-choice suite whose prompt
// lengths are a permutation of its case order, with each case's length.
func replayFixtureSuite(t *testing.T, cases int) ([]byte, map[string]int) {
	t.Helper()
	suite := evaluation.MultipleChoiceSuite{
		Kind: evaluation.MultipleChoiceKind, Schema: "fixture/replay", Source: "store/mmlu",
		Normalization: sequencescore.NormalizationSum, Aggregation: evaluation.AggregationAccuracy,
	}
	lengths := map[string]int{}
	for index := range cases {
		name := fmt.Sprintf("case-%03d", index)
		prompt := fmt.Sprintf("Question %d:%s\nAnswer:", index, strings.Repeat(" word", (index*37)%cases+1))
		suite.Cases = append(suite.Cases, evaluation.MultipleChoiceCase{Name: name, Prompt: prompt, Candidates: []string{" A", " B", " C", " D"}, Answer: index % 4})
		lengths[name] = len(prompt)
	}
	data, err := json.Marshal(suite)
	if err != nil {
		t.Fatal(err)
	}
	return data, lengths
}

// TestEvaluationReplayCarriesIdenticalOutputs holds a re-evaluation to a
// replay first. In validate, every benchmark that runs replays its newest
// record, whatever made it run (a moved surface, a failed run, a stale
// activation), and one no record measures runs in full; evaluate decides
// whether the replay carries. In the evaluation, a replay of 32 length-spread
// cases that reproduces every output carries the record forward as a run at
// the new commit, which the evidence index reads as it reads a fresh one and
// marks carried, naming the replay; a carry of a carry still stands on the
// full report; one case whose floats moved by one unit in the last place runs
// the suite in full; and a plan that differs beyond the commit (another
// environment, another dataset) or a hosted environment runs in full without
// a replay. The replay spans the suite's prompt lengths and includes its
// longest prompts.
func TestEvaluationReplayCarriesIdenticalOutputs(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, value string) artifact.ID { return testutil.ArtifactID(t, kind, value) }
	model, definition, recipeID := id(artifact.KindModel, "model"), id(artifact.KindModelDefinition, "definition"), id(artifact.KindRecipe, "recipe")
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{Key: "fixture/model", Artifacts: []artifact.Descriptor{{ID: model}, {ID: definition}, {ID: recipeID}}}); err != nil {
		t.Fatal(err)
	}
	environment := func(backend, driver string) runrecord.Environment {
		value, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: backend, Driver: driver})
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	device := environment("fixture", "fixture")
	data, lengths := replayFixtureSuite(t, 100)
	var suiteCases struct {
		Cases []evaluation.MultipleChoiceCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &suiteCases); err != nil {
		t.Fatal(err)
	}
	scored := 0
	var samples []string
	progress := evaluation.WithProgress(ctx, func(step evaluation.Progress) {
		if step.Sample != "" {
			samples = append(samples, step.Sample)
		}
	})
	// evaluate evaluates the fixture suite at one commit, carrying previous
	// forward when it is valid, and says how many cases the runtime scored.
	evaluate := func(commit string, platform runrecord.Environment, drift string, suiteData []byte, previous artifact.ID) (evaluation.CampaignResult, evaluation.CompiledSuite, int) {
		t.Helper()
		scored, samples = 0, nil
		campaign, err := evaluation.NewIsolatedCampaign(store, replayFixtureRuntime{scored: &scored, drift: drift},
			modelrecipe.ProgramIdentity{Model: model, Definition: definition, Recipe: recipeID}, platform, strings.Repeat(commit, 40))
		if err != nil {
			t.Fatal(err)
		}
		suite, err := evaluation.CompileSuite(suiteData, campaign.Authorities())
		if err != nil {
			t.Fatal(err)
		}
		result, err := campaign.EvaluateOrCarry(progress, suite, previous)
		if err != nil {
			t.Fatal(err)
		}
		return result, suite, scored
	}
	latest := func(suite evaluation.CompiledSuite) evaluation.EvalSummary {
		t.Helper()
		index := evaluation.LatestEvidence(ctx, store, map[artifact.ID]string{suite.Descriptor().Dataset: "store/mmlu"}, 0)
		summaries := index.EvaluationsByRecipe[recipeID]
		if len(summaries) != 1 {
			t.Fatalf("the evidence index holds %d summaries of the suite, want 1: %+v", len(summaries), summaries)
		}
		return summaries[0]
	}
	receipt := func(report artifact.ID) evaluation.ReplayReceipt {
		t.Helper()
		content, found, err := artifact.ReadContent(ctx, store, report)
		var value evaluation.ReplayReceipt
		if err == nil && found {
			err = json.Unmarshal(content.Data, &value)
		}
		if err != nil || !found {
			t.Fatalf("replay receipt %s is unreadable: found=%t %v", report, found, err)
		}
		return value
	}

	full, suite, cases := evaluate("a", device, "", data, artifact.ID{})
	if cases != 100 || full.Carried.Valid() {
		t.Fatalf("the first evaluation scored %d of 100 cases, carried %s", cases, full.Carried)
	}

	// A replay that reproduces every output carries the record forward.
	carried, suite, cases := evaluate("b", device, "", data, full.Evaluation)
	if cases != 32 || carried.Carried != full.Evaluation || !slices.Equal(carried.Metrics, full.Metrics) {
		t.Fatalf("an identical replay scored %d cases and carried %s with %v, want 32 cases carrying %s with %v; samples %q",
			cases, carried.Carried, carried.Metrics, full.Evaluation, full.Metrics, samples)
	}
	summary := latest(suite)
	if summary.Record != carried.Evaluation || summary.Carried != carried.Report || summary.Metrics["accuracy"] != full.Metrics[0].Value ||
		summary.Prompting != evaluation.PromptingRawCompletion.Label() || !summary.Reproducible {
		t.Fatalf("the evidence index does not read the carried record as a fresh one marked carried: %+v", summary)
	}
	if _, err := evaluation.RequireEvaluationEvidence(ctx, store, carried.Evidence); err != nil {
		t.Fatalf("the carried evaluation's evidence does not verify: %v", err)
	}
	run, err := runrecord.RequireRun(ctx, store, carried.Run)
	if err != nil || run.CodeCommit != strings.Repeat("b", 40) || run.Outcome != runrecord.OutcomeSucceeded || !slices.Equal(run.Outputs, []artifact.ID{carried.Report}) {
		t.Fatalf("the carried run is not a succeeded run at the new commit reporting its receipt: %+v %v", run, err)
	}
	named := receipt(carried.Report)
	if named.Plan != suite.Plan().Identity() || named.Carried != full.Evaluation || named.Report != full.Report || !named.Replay.Valid() ||
		named.Replay == full.Report || len(named.Cases) != 32 {
		t.Fatalf("the receipt does not name the plan, the record carried, its report, the replay and 32 cases: %+v", named)
	}

	// The replay spans the suite's prompt lengths and includes its longest.
	var replayed []string
	for _, index := range named.Cases {
		replayed = append(replayed, suiteCases.Cases[index].Name)
	}
	byLength := slices.SortedFunc(maps.Keys(lengths), func(left, right string) int {
		return cmp.Or(cmp.Compare(lengths[left], lengths[right]), strings.Compare(left, right))
	})
	for _, longest := range byLength[len(byLength)-8:] {
		if !slices.Contains(replayed, longest) {
			t.Errorf("the replay omits %s, one of the 8 longest prompts", longest)
		}
	}
	for quartile := range 4 {
		if !slices.ContainsFunc(byLength[quartile*25:(quartile+1)*25], func(name string) bool { return slices.Contains(replayed, name) }) {
			t.Errorf("the replay takes no case from length quartile %d: %v", quartile+1, replayed)
		}
	}
	if !slices.Contains(replayed, byLength[0]) {
		t.Errorf("the replay omits the shortest prompt %s", byLength[0])
	}

	// A carry of a carry still compares against, and stands on, the full report.
	again, _, cases := evaluate("c", device, "", data, carried.Evaluation)
	if cases != 32 || again.Carried != carried.Evaluation || receipt(again.Report).Report != full.Report {
		t.Fatalf("a replay against a carried record scored %d cases, carried %s on %s", cases, again.Carried, receipt(again.Report).Report)
	}

	// One case whose scores moved by one unit in the last place runs in full.
	longest := slices.IndexFunc(suiteCases.Cases, func(c evaluation.MultipleChoiceCase) bool { return c.Name == byLength[len(byLength)-1] })
	drifted, suite, cases := evaluate("d", device, suiteCases.Cases[longest].Prompt, data, again.Evaluation)
	if cases != 32+100 || drifted.Carried.Valid() || latest(suite).Carried.Valid() || latest(suite).Record != drifted.Evaluation ||
		!slices.ContainsFunc(samples, func(sample string) bool { return strings.Contains(sample, fmt.Sprintf("case %d replayed as", longest)) }) {
		t.Fatalf("a replay that moved one case scored %d cases (want 32+100) and carried %s; samples %q", cases, drifted.Carried, samples)
	}

	// A plan that differs beyond the commit, or a hosted environment, runs
	// in full without a replay.
	beyond := func(sample string) bool { return strings.Contains(sample, "differs from this one beyond the commit") }
	other, _, cases := evaluate("e", environment("fixture", "other"), "", data, drifted.Evaluation)
	if cases != 100 || other.Carried.Valid() || !slices.ContainsFunc(samples, beyond) {
		t.Fatalf("another environment scored %d cases (want 100, no replay) and carried %s; samples %q", cases, other.Carried, samples)
	}
	fewer, _ := replayFixtureSuite(t, 99)
	reimported, _, cases := evaluate("f", environment("fixture", "other"), "", fewer, other.Evaluation)
	if cases != 99 || reimported.Carried.Valid() || !slices.ContainsFunc(samples, beyond) {
		t.Fatalf("another dataset scored %d cases (want 99, no replay) and carried %s; samples %q", cases, reimported.Carried, samples)
	}
	remote := environment(runrecord.BackendRemote, "fixture")
	hosted, _, _ := evaluate("1", remote, "", data, artifact.ID{})
	if hostedAgain, _, cases := evaluate("2", remote, "", data, hosted.Evaluation); cases != 100 || hostedAgain.Carried.Valid() {
		t.Fatalf("a hosted environment scored %d cases (want 100, no replay) and carried %s", cases, hostedAgain.Carried)
	}

	// Validate replays the newest record of every benchmark it runs.
	root, beforeSurface, beforeDocs, _ := surfaceHistory(t)
	directories, err := surfaceDirectories(ctx, root, modalitySurfaces["evaluation"].roots)
	if err != nil {
		t.Fatal(err)
	}
	benchmark := func(name string) ModelValidation {
		return ModelValidation{
			Model: id(artifact.KindModel, name), ModelName: name, Location: "C:/models/" + name + "/m.gguf",
			Kind: recipe.TaskInference, Validation: "mmlu-pro", Surface: "evaluation", Recipe: id(artifact.KindRecipe, name),
			Protocol: evaluation.PromptingRawCompletion.Label(),
		}
	}
	ran := func(commit string, outcome runrecord.Outcome) boundRecord {
		return boundRecord{run: runrecord.Run{Outcome: outcome, CodeCommit: commit}, found: true, replay: full.Evaluation}
	}
	moved, current, failed, unmeasured, stale := benchmark("Moved"), benchmark("Current"), benchmark("Failed"), benchmark("Unmeasured"), benchmark("Stale")
	stale.Stale = "its recipe was replaced"
	records := map[string]boundRecord{
		moved.ModelName: ran(beforeSurface, runrecord.OutcomeSucceeded), current.ModelName: ran(beforeDocs, runrecord.OutcomeSucceeded),
		failed.ModelName: ran(beforeDocs, runrecord.OutcomeFailed), unmeasured.ModelName: {}, stale.ModelName: ran(beforeSurface, runrecord.OutcomeSucceeded),
	}
	// The cells are judged as judgeBound judges them, from records read here
	// rather than from a store.
	read := acceptedRecords{bound: map[boundKey]error{}}
	var cells []ModelValidation
	for _, cell := range []ModelValidation{moved, current, failed, unmeasured, stale} {
		record := records[cell.ModelName]
		cell.Protocol, cell.Cost, cell.Replay = record.protocol, record.cost, record.replay
		read.bound[boundKey{cell.Recipe, cell.Validation, cell.Protocol}] = recordCurrent(ctx, root, directories, cell, record)
		cells = append(cells, cell)
	}
	plan := SelectValidation(Inputs{Registered: cells, Current: read.current})
	replays := map[string]bool{}
	for _, selected := range append(plan.Run, plan.Background...) {
		_, args, err := reacquireCommand(selected.ModelValidation, "C:/stores/selected")
		at := slices.Index(args, "-replay")
		if err != nil || at >= 0 && (at+1 == len(args) || args[at+1] != full.Evaluation.String()) {
			t.Fatalf("%s bound %v, %v", selected.ModelName, args, err)
		}
		replays[selected.ModelName] = at >= 0
	}
	if want := map[string]bool{moved.ModelName: true, failed.ModelName: true, unmeasured.ModelName: false, stale.ModelName: true}; fmt.Sprint(replays) != fmt.Sprint(want) {
		t.Fatalf("the benchmarks that run replay %v, want %v", replays, want)
	}
	if len(plan.Reuse) != 1 || plan.Reuse[0].ModelName != current.ModelName {
		t.Fatalf("the benchmark current at this tree was not reused: %+v", plan.Reuse)
	}
}
