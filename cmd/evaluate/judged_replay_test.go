package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// answeringJudge is a judged runtime whose answering alone stands in for the judge's model: it binds a picture as the
// judge does, against the render that drew it, answers every question "A", sure of it, and counts the questions.
type answeringJudge struct {
	judgedRuntime
	asked *int
}

// ChoiceTokens names letter A token 0 and letter B token 1.
func (answeringJudge) ChoiceTokens(choices []string) ([]tokenizer.TokenID, error) {
	return []tokenizer.TokenID{0, 1}[:len(choices)], nil
}

func (j answeringJudge) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	*j.asked++
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 1})
	}
	return make([]tokenizer.TokenID, 2), "A", options.OnToken(inference.TokenEvent{Piece: "A", Logits: []float32{2, 0}})
}

// TestAJudgedAnswerIsReplayedForTheSameInputs holds a judged session's answers to being kept (evaluation holds the
// suite's side): a judged session's campaign, kept as openJudgedSession keeps it, judges a picture the render manifest
// binds to its recipe; judged again at the same commit, the suite puts no question to the judge and scores the same.
func TestAJudgedAnswerIsReplayedForTheSameInputs(t *testing.T) {
	t.Parallel()
	model := testutil.ArtifactID(t, artifact.KindModel, "drawing-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "drawing-recipe")
	directory, key, store := renderedDirectory(t, model, recipeID)
	definition := testutil.ArtifactID(t, artifact.KindModelDefinition, "drawing-definition")
	testcheck.Must(store.Commit(t.Context(), artifact.Batch{Key: "fixture/judged", Artifacts: []artifact.Descriptor{{ID: model}, {ID: definition}, {ID: recipeID}}}))(t)
	environment := testcheck.Must(runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"}))(t)
	items := []evaluation.TIFAItem{{ID: "coco_1", Caption: "a cat", Questions: []evaluation.TIFAQuestion{
		{Question: "is there a cat?", Choices: []string{"yes", "no"}, Answer: "yes"},
		{Question: "is the cat asleep?", Choices: []string{"yes", "no"}, Answer: "no"},
	}}}
	judgedOnce := func() (int, evaluation.CampaignResult) {
		t.Helper()
		asked := 0
		judge := answeringJudge{judgedRuntime{projectionRuntime: projectionRuntime{mediaFiles: &mediaFiles{root: directory}, identity: evaluation.MediaRuntime{Model: model, Recipe: recipeID}},
			model: model, recipe: recipeID, store: store}, &asked}
		campaign := keepJudgedAnswers(testcheck.Must(evaluation.NewIsolatedCampaign(store, judge, modelrecipe.ProgramIdentity{Model: model, Definition: definition, Recipe: recipeID},
			environment, strings.Repeat("a", 40)))(t))
		suite := testcheck.Must(evaluation.BuildTIFAQuestions(items, map[string]string{"coco_1": key}, judge.Judge()))(t)
		compiled := testcheck.Must(campaign.CompileSuite(t.Context(), testcheck.Must(json.Marshal(suite))(t)))(t)
		return asked, testcheck.Must(campaign.Evaluate(t.Context(), compiled))(t)
	}
	asked, first := judgedOnce()
	if asked != 2 {
		t.Fatalf("the first judgement put %d questions to the judge, want both", asked)
	}
	if asked, again := judgedOnce(); asked != 0 || len(again.Metrics) == 0 || again.Metrics[0] != first.Metrics[0] {
		t.Fatalf("judged again, the suite put %d questions and read %v; want none, and %v", asked, again.Metrics, first.Metrics)
	}
}
