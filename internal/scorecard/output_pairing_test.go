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
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// looker is a judge that answers every question "A", right, except those asked of the pictures it answers wrong.
type looker struct{ wrong map[string]bool }

func (judge looker) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	answer := "A"
	for picture := range judge.wrong {
		if strings.Contains(prompt, picture) {
			answer = "B"
		}
	}
	// The prompt reads as one token and the answer as one, streamed as a model streams it.
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 1})
	if err := options.OnToken(inference.TokenEvent{Piece: answer}); err != nil {
		return nil, "", err
	}
	return make([]tokenizer.TokenID, len("A")+1), "", nil
}

func (looker) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("the fixture answers questions and scores no candidates")
}

// TestAQualityVerdictCountsMediaNotQuestions holds a decline of a judged quality to the pictures it rests on: the
// several questions a suite asks of one picture are one piece of evidence of how well it was drawn, and on 2026-11-09
// the scorecard counted them one by one (the owner's review). A suite of 10 pictures, 2 questions each, is answered
// right throughout, then wrong on every question of 3 pictures: 6 questions of 20 went wrong, a decline counted by
// question (chance 0.0156), and 3 pictures of 10, none counted by picture (0.125); asking each question twice reads
// the same verdict. Wrong on 6 pictures, the decline stands (0.0156).
func TestAQualityVerdictCountsMediaNotQuestions(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	const pictures = 10
	for _, asked := range []int{2, 4} {
		root := filepath.Join(t.TempDir(), "store")
		store := testcheck.Must(overgodb.Create(root))(t)
		identity := modelrecipe.ProgramIdentity{
			Model:      testutil.ArtifactID(t, artifact.KindModel, "judged model"),
			Definition: testutil.ArtifactID(t, artifact.KindModelDefinition, "judged definition"),
			Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, "judged recipe"),
		}
		testcheck.Must(store.Commit(ctx, artifact.Batch{Key: "fixture/judged", Artifacts: []artifact.Descriptor{{ID: identity.Model}, {ID: identity.Definition}, {ID: identity.Recipe}}}))(t)
		environment := testcheck.Must(runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"}))(t)
		suite := evaluation.GeneratedAnswerSuite{Kind: evaluation.GeneratedAnswerKind, Schema: "fixture/judged/v1", Source: "judged", Transforms: []string{evaluation.TransformTrimSpace}}
		for picture := range pictures {
			for question := range asked {
				suite.Cases = append(suite.Cases, evaluation.GeneratedAnswerCase{
					Name:   fmt.Sprintf("judged/picture-%d/%d", picture, question),
					Prompt: fmt.Sprintf("picture %d: question %d?", picture, question), MaxTokens: 1, Answers: []string{"A"},
					Group: fmt.Sprintf("picture-%d", picture),
				})
			}
		}
		source := testcheck.Must(json.Marshal(suite))(t)
		cases := uint64(len(suite.Cases))
		contract := Contract{Metric: shareMetric, Origin: originEvaluation, Suite: "judged", Protocol: "chat-template", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "judged comparison")}
		// read answers the suite at a commit of its own, wrong on every question of the first wrong pictures.
		read := func(commit string, wrong int) Cell {
			t.Helper()
			missed := map[string]bool{}
			for picture := range wrong {
				missed[fmt.Sprintf("picture %d:", picture)] = true
			}
			campaign := testcheck.Must(evaluation.NewCampaign(store, looker{missed}, identity, environment, strings.Repeat(commit, 40)))(t)
			result := testcheck.Must(campaign.Evaluate(ctx, testcheck.Must(evaluation.CompileSuite(source, campaign.Authorities()))(t)))(t)
			value := float64(int(cases)-wrong*asked) / float64(cases)
			return Cell{Value: value, Source: result.Evaluation, Direction: runrecord.DirectionMaximize, Cases: cases, Noise: shareError(value, cases), Contract: contract,
				Compared: "internal/fixture TestAQualityVerdictCountsMediaNotQuestions"}
		}
		baseline, three, six := read("a", 0), read("b", 3), read("c", 6)
		testcheck.NoError(t, store.Close())
		row := func(quality Cell) Row {
			cells := map[metric.Name]Cell{}
			for _, column := range metric.Columns {
				cells[column] = inapplicable("not measured in this fixture")
			}
			cells[metric.Quality] = quality
			return Row{Model: identity.Model, Name: "judged", Task: recipe.TaskImageGen, Cells: cells}
		}
		for _, moved := range []struct {
			now    Cell
			stands bool
			says   string
		}{
			{three, false, "3 of the 10 outputs both readings answered questions of had fewer of their questions answered right and 0 more, and so many go fewer with chance 0.125 "},
			{six, true, "6 of the 10 outputs both readings answered questions of had fewer of their questions answered right and 0 more, and so many go fewer with chance 0.0156 "},
		} {
			gaps := Record{Commit: "now", Rows: []Row{row(moved.now)}}.Gaps(&Record{Commit: "then", Rows: []Row{row(baseline)}})
			if len(gaps) != 1 || gaps[0].quality == nil {
				t.Fatalf("%d questions a picture: the move from %g to %g files %+v, want the one decline of its quality", asked, baseline.Value, moved.now.Value, gaps)
			}
			kept, dropped := Supported(ctx, root, gaps)
			switch {
			case moved.stands && (len(kept) != 1 || !strings.Contains(kept[0].Detail, moved.says)):
				t.Errorf("%d questions a picture: kept %+v and dropped %q, want the decline filed saying %q", asked, kept, dropped, moved.says)
			case !moved.stands && (len(kept) != 0 || len(dropped) != 1 || !strings.Contains(dropped[0], moved.says)):
				t.Errorf("%d questions a picture: kept %+v and dropped %q, want the decline dropped saying %q", asked, kept, dropped, moved.says)
			}
		}
	}
}
