package scorecard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard/metric"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// answerer is a model that scores the first of a case's two candidates the
// likelier, except on the prompts it answers wrong.
type answerer struct{ wrong map[string]bool }

func (answerer) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("the fixture scores candidates and writes nothing")
}

func (model answerer) ScoreContinuations(_ context.Context, prompt string, _ []string) ([]sequencescore.Score, error) {
	scores := []sequencescore.Score{{LogProbability: -1, Tokens: 1}, {LogProbability: -2, Tokens: 1}}
	if model.wrong[prompt] {
		scores[0], scores[1] = scores[1], scores[0]
	}
	return scores, nil
}

// TestAQualityMoveIsJudgedByTheCasesAnsweredDifferently holds a decline of a
// held-out quality to what its two readings rest on. One model answers one
// suite of 40 cases at four commits into a temporary store, and each later
// reading is set against the first, 30 cases right, case by case. One case
// answered wrong that was answered right moves the share by more than the
// scorecard prints, and is no decline. Eight answered wrong are one. Eight
// answered wrong and four answered right that were wrong move the share by four
// cases, and is none either: so many answered differently go both ways by
// chance. Readings whose reports the store does not hold are judged by their
// stated standard errors, a move within them filing nothing; a reading with
// none, or a store that cannot be opened, is judged by its value, as it was.
func TestAQualityMoveIsJudgedByTheCasesAnsweredDifferently(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	identity := modelrecipe.ProgramIdentity{
		Model:      testutil.ArtifactID(t, artifact.KindModel, "paired model"),
		Definition: testutil.ArtifactID(t, artifact.KindModelDefinition, "paired definition"),
		Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, "paired recipe"),
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "fixture/paired", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	const cases = 40
	suite := evaluation.MultipleChoiceSuite{Kind: evaluation.MultipleChoiceKind, Schema: "fixture/paired/v1", Source: "paired", Normalization: sequencescore.NormalizationSum, Aggregation: evaluation.AggregationAccuracy}
	for index := range cases {
		suite.Cases = append(suite.Cases, evaluation.MultipleChoiceCase{Name: fmt.Sprintf("case-%02d", index), Prompt: fmt.Sprintf("Question %d\nAnswer:", index), Candidates: []string{" A", " B"}})
	}
	source, err := json.Marshal(suite)
	if err != nil {
		t.Fatal(err)
	}
	contract := Contract{Metric: shareMetric, Origin: originEvaluation, Suite: "paired", Protocol: "raw-completion", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "paired comparison")}
	// read answers the suite at a commit of its own, so each case is put to the model again, wrong on the cases from first up to last, and returns the cell the scorecard reads.
	read := func(commit string, first, last int) Cell {
		t.Helper()
		wrong := map[string]bool{}
		for _, testCase := range suite.Cases[first:last] {
			wrong[testCase.Prompt] = true
		}
		campaign, err := evaluation.NewCampaign(store, answerer{wrong}, identity, environment, strings.Repeat(commit, 40))
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := evaluation.CompileSuite(source, campaign.Authorities())
		if err != nil {
			t.Fatal(err)
		}
		result, err := campaign.Evaluate(ctx, compiled)
		if err != nil {
			t.Fatal(err)
		}
		value := float64(cases-(last-first)) / cases
		if len(result.Metrics) != 1 || result.Metrics[0].Value != value {
			t.Fatalf("the reading at %s measured %+v, want an accuracy of %g", commit, result.Metrics, value)
		}
		return Cell{Value: value, Source: result.Evaluation, Direction: runrecord.DirectionMaximize, Cases: cases, Noise: shareError(value, cases), Contract: contract,
			Compared: "internal/fixture TestPairedPathMatchesItsReference"}
	}
	baseline := read("a", 0, 10)
	oneLost := read("b", 0, 11)
	eightLost := read("c", 0, 18)
	bothWays := read("d", 4, 18)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	row := func(quality Cell) Row {
		cells := map[metric.Name]Cell{}
		for _, column := range metric.Columns {
			cells[column] = inapplicable("not measured in this fixture")
		}
		cells[metric.Quality] = quality
		return Row{Model: identity.Model, Name: "paired", Task: recipe.TaskInference, Cells: cells}
	}
	// judge files the move from then to now as Gaps does, which reads any move the scorecard prints as a decline, and judges it at root.
	judge := func(root string, then, now Cell) ([]Gap, []string) {
		t.Helper()
		gaps := Record{Commit: "now", Rows: []Row{row(now)}}.Gaps(&Record{Commit: "then", Rows: []Row{row(then)}})
		if len(gaps) != 1 || gaps[0].quality == nil {
			t.Fatalf("the move from %g to %g files %+v, want the one decline of its quality", then.Value, now.Value, gaps)
		}
		return Supported(ctx, root, gaps)
	}
	for _, moved := range []struct {
		name   string
		now    Cell
		stands bool
		paired string
	}{
		{"one case answered wrong", oneLost, false, "1 of the 40 cases both readings answered went from right to wrong and 0 from wrong to right, and so many go wrong with chance 0.5 "},
		{"eight cases answered wrong", eightLost, true, "8 of the 40 cases both readings answered went from right to wrong and 0 from wrong to right, and so many go wrong with chance 0.00391 "},
		{"eight answered wrong and four right", bothWays, false, "8 of the 40 cases both readings answered went from right to wrong and 4 from wrong to right, and so many go wrong with chance 0.194 "},
	} {
		kept, dropped := judge(root, baseline, moved.now)
		switch {
		case moved.stands && (len(kept) != 1 || len(dropped) != 0 || !strings.Contains(kept[0].Detail, "- the decline stands because "+moved.paired)):
			t.Errorf("%s: kept %+v and dropped %q, want the decline filed with what the cases say", moved.name, kept, dropped)
		case !moved.stands && (len(kept) != 0 || len(dropped) != 1 || !strings.Contains(dropped[0], "is not filed because "+moved.paired)):
			t.Errorf("%s: kept %+v and dropped %q, want the decline dropped with what the cases say", moved.name, kept, dropped)
		}
	}

	// unstored is a reading no stored record names, with the value and standard error given.
	unstored := func(name string, value, noise float64) Cell {
		cell := baseline
		cell.Source, cell.Value, cell.Noise = testutil.ArtifactID(t, artifact.KindEvaluation, name), value, noise
		return cell
	}
	then := unstored("then", 0.75, 0.07)
	for _, fallback := range []struct {
		name         string
		root         string
		then, now    Cell
		stands       bool
		judgedBy     string
		withoutCases string
	}{
		{"within its noise", root, then, unstored("within", 0.7, 0.07), false, "so its move of 0.05 is judged against 2 times 0.099", "the store names no outcome of a case both readings answered"},
		{"beyond its noise", root, then, unstored("beyond", 0.3, 0.07), true, "so its move of 0.45 is judged against 2 times 0.099", "the store names no outcome of a case both readings answered"},
		{"with no noise", root, unstored("then", 0.75, 0), unstored("plain", 0.7, 0), true, "no standard error is stated, so it is judged by its value", "the store names no outcome of a case both readings answered"},
		{"with no store", "", baseline, eightLost, false, "so its move of 0.2 is judged against 2 times 0.104", "the store cannot be opened"},
		{"with no store and no noise", "", unstored("then", 0.75, 0), unstored("plain", 0.55, 0), true, "no standard error is stated, so it is judged by its value", "the store cannot be opened"},
	} {
		kept, dropped := judge(fallback.root, fallback.then, fallback.now)
		said, filed := strings.Join(dropped, "\n"), len(kept) == 1 && len(dropped) == 0
		if filed {
			said = kept[0].Detail
		}
		if filed != fallback.stands || len(kept)+len(dropped) != 1 || !strings.Contains(said, fallback.judgedBy) || !strings.Contains(said, fallback.withoutCases) {
			t.Errorf("a decline %s: kept %+v and dropped %q, want it filed %t, judged %q because %q", fallback.name, kept, dropped, fallback.stands, fallback.judgedBy, fallback.withoutCases)
		}
	}
}
