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
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
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
	scored      *int
	drift       string
	first, last *time.Duration
	cancel      context.CancelFunc
	failFirst   bool
}

func (replayFixtureRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("the replay fixture scores likelihoods only")
}

func (runtime replayFixtureRuntime) ScoreContinuations(_ context.Context, prompt string, candidates []string) ([]sequencescore.Score, error) {
	*runtime.scored++
	if runtime.first != nil {
		now, err := processmeasure.Counter()
		if err != nil {
			return nil, err
		}
		if *runtime.scored == 1 {
			*runtime.first = now
		}
		*runtime.last = now
	}
	if runtime.cancel != nil {
		runtime.cancel()
		return nil, context.Canceled
	}
	if runtime.failFirst && *runtime.scored == 1 {
		return nil, errors.New("one replay execution failed")
	}
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
// replay first. In validate, every benchmark runs every cycle and replays its
// newest record, whatever else is true of it (a stale activation, a failed
// run), and one no record measures runs in full; evaluate decides whether the
// replay carries. In the evaluation, a replay of 32 length-spread
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
	var first, last time.Duration
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
		campaign, err := evaluation.NewIsolatedCampaign(store, replayFixtureRuntime{scored: &scored, drift: drift, first: &first, last: &last},
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
	if summary.CodeCommit != strings.Repeat("b", 40) || summary.ScoredCommit != strings.Repeat("a", 40) {
		t.Fatalf("a carried result reads commit %q scored at %q, want the carrying run's %q and the scoring run's %q", summary.CodeCommit, summary.ScoredCommit, strings.Repeat("b", 40), strings.Repeat("a", 40))
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
	legacyID, err := artifact.JSONID(artifact.KindEvaluation, struct {
		Version uint16      `json:"version"`
		Plan    artifact.ID `json:"plan"`
		Carried artifact.ID `json:"carried"`
		Report  artifact.ID `json:"report"`
		Replay  artifact.ID `json:"replay"`
		Cases   []int       `json:"cases"`
	}{named.Version, named.Plan, named.Carried, named.Report, named.Replay, named.Cases})
	if err != nil || legacyID != carried.Report {
		t.Fatalf("existing carried receipt content identity changed: %s versus%s: %v", legacyID, carried.Report, err)
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
	if summary := latest(suite); summary.CodeCommit != strings.Repeat("c", 40) || summary.ScoredCommit != strings.Repeat("a", 40) {
		t.Fatalf("a carry of a carry reads commit %q scored at %q, want the third run's and the first run's %q", summary.CodeCommit, summary.ScoredCommit, strings.Repeat("a", 40))
	}

	// A rerun at the commit just evaluated, as a cycle resumed at an unchanged HEAD does, is not refused
	// where the first run bound the plan's campaign: the plan's campaign stays the first run's.
	campaignBefore, found, err := artifact.ResolveAlias(ctx, store, "evaluation/campaigns/"+suite.Plan().Identity().String())
	if err != nil || !found {
		t.Fatalf("the plan's campaign is bound to %s (found %t, %v), want the carry of a carry's", campaignBefore, found, err)
	}
	if rerun, _, _ := evaluate("c", device, "", data, again.Evaluation); !rerun.Report.Valid() {
		t.Fatalf("a rerun at the same commit reported %s", rerun.Report)
	}
	if campaignAfter, _, err := artifact.ResolveAlias(ctx, store, "evaluation/campaigns/"+suite.Plan().Identity().String()); err != nil || campaignAfter != campaignBefore {
		t.Fatalf("a rerun at the same commit moved the plan's campaign from %s to %s (%v)", campaignBefore, campaignAfter, err)
	}

	// One case whose scores moved by one unit in the last place runs in full.
	longest := slices.IndexFunc(suiteCases.Cases, func(c evaluation.MultipleChoiceCase) bool { return c.Name == byLength[len(byLength)-1] })
	drifted, suite, cases := evaluate("d", device, suiteCases.Cases[longest].Prompt, data, again.Evaluation)
	if cases != 32+100 || drifted.Carried.Valid() || latest(suite).Carried.Valid() || latest(suite).Record != drifted.Evaluation ||
		!slices.ContainsFunc(samples, func(sample string) bool { return strings.Contains(sample, fmt.Sprintf("case %d replayed as", longest)) }) {
		t.Fatalf("a replay that moved one case scored %d cases (want 32+100) and carried %s; samples %q", cases, drifted.Carried, samples)
	}
	driftRun, err := runrecord.RequireRun(ctx, store, drifted.Run)
	if err != nil || driftRun.MeasuredNS < uint64(last-first) {
		t.Errorf("fallback wall %d loses executed replay/fallback span %d: %v", driftRun.MeasuredNS, last-first, err)
	}
	if wall, known := drifted.Resources.Measure(runrecord.ResourceWallNS); !known || wall != driftRun.MeasuredNS {
		t.Errorf("fallback resources lose attempt wall: %+v", drifted.Resources)
	}
	fallback := receipt(drifted.Report)
	if fallback.Carried.Valid() || !fallback.Report.Valid() || !fallback.Replay.Valid() || len(fallback.Cases) != 32 {
		t.Errorf("fallback loses its fresh full report and32 replay cases: %+v", fallback)
	}

	// Validate re-acquires every benchmark and replays the newest record of
	// each recipe that has one, whatever else is true of it: its run at the
	// newest commit and a stale activation alike. One no record measures runs
	// in full.
	catalog := benchmarkEvidence{suites: []evaluation.CompiledSuite{suite}, evidence: evaluation.LatestEvidence(ctx, store, map[artifact.ID]string{suite.Descriptor().Dataset: "store/mmlu"}, 0)}
	mmlu := func(name string, recipeID artifact.ID) ModelValidation {
		return ModelValidation{
			Model: model, ModelName: name, Location: "C:/models/" + name + "/m.gguf",
			Kind: recipe.TaskInference, Validation: "mmlu", Recipe: recipeID,
		}
	}
	measured, unmeasured, stale := mmlu("Measured", recipeID), mmlu("Unmeasured", id(artifact.KindRecipe, "unmeasured")), mmlu("Stale", recipeID)
	stale.Stale = "its recipe was replaced"
	var cells []ModelValidation
	for _, cell := range []ModelValidation{measured, unmeasured, stale} {
		bound, err := catalog.bind(ctx, store, cell)
		if err != nil {
			t.Fatal(err)
		}
		cells = append(cells, bound)
	}
	plan := SelectValidation(Inputs{Registered: cells})
	replays := map[string]string{}
	for _, selected := range slices.Concat(plan.Run, plan.Background) {
		_, args, err := reacquireCommand(selected.ModelValidation, "C:/stores/selected")
		if err != nil {
			t.Fatalf("%s bound %v, %v", selected.ModelName, args, err)
		}
		if at := slices.Index(args, "-replay"); at >= 0 {
			replays[selected.ModelName] = args[at+1]
		}
	}
	if want := (map[string]string{measured.ModelName: drifted.Evaluation.String(), stale.ModelName: drifted.Evaluation.String()}); !maps.Equal(replays, want) {
		t.Fatalf("the benchmarks that run replay %v, want %v", replays, want)
	}

	t.Run("cancelled replay retains spent work", func(t *testing.T) {
		cancelledCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		calls := 0
		campaign, err := evaluation.NewIsolatedCampaign(store, replayFixtureRuntime{scored: &calls, cancel: cancel},
			modelrecipe.ProgramIdentity{Model: model, Definition: definition, Recipe: recipeID}, device, strings.Repeat("0", 40))
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := evaluation.CompileSuite(data, campaign.Authorities())
		if err != nil {
			t.Fatal(err)
		}
		result, evaluateErr := campaign.EvaluateOrCarry(cancelledCtx, compiled, again.Evaluation)
		terminal, err := runrecord.RequireRun(ctx, store, result.Run)
		wall, known := result.Resources.Measure(runrecord.ResourceWallNS)
		if !errors.Is(evaluateErr, context.Canceled) || err != nil || terminal.Outcome != runrecord.OutcomeCancelled ||
			calls != 1 || !known || wall == 0 || wall != terminal.MeasuredNS || result.Resources.Scope.Attempt != result.Run {
			t.Fatalf("cancelled replay: calls%d, run%+v, resources%+v, error%v/read%v", calls, terminal, result.Resources, evaluateErr, err)
		}
	})
	t.Run("partial replay remains an unknown denominator", func(t *testing.T) {
		calls := 0
		campaign, err := evaluation.NewIsolatedCampaign(store, replayFixtureRuntime{scored: &calls, failFirst: true},
			modelrecipe.ProgramIdentity{Model: model, Definition: definition, Recipe: recipeID}, device, strings.Repeat("9", 40))
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := evaluation.CompileSuite(data, campaign.Authorities())
		if err != nil {
			t.Fatal(err)
		}
		result, err := campaign.EvaluateOrCarry(ctx, compiled, again.Evaluation)
		if err != nil || result.Carried.Valid() || calls != 101 {
			t.Fatalf("partial replay result=%+v, calls%d, error%v", result, calls, err)
		}
		scope, err := evaluation.RequireReplayReceipt(ctx, store, result.Report)
		if err != nil || !scope.Fallback || scope.Carried.Valid() || !scope.Report.Valid() || scope.Replay.Valid() || len(scope.Cases) != 0 {
			t.Fatalf("partial replay scope=%+v: %v", scope, err)
		}
		if _, err := evaluation.RequireEvaluationEvidence(ctx, store, result.Evidence); err != nil {
			t.Fatalf("fallback evidence no longer verifies: %v", err)
		}
	})

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
}
