package evaluation

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// countedJudge is a choiceJudge that counts the questions put to it.
type countedJudge struct {
	choiceJudge
	asked *int
}

func (g countedJudge) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	*g.asked++
	return g.choiceJudge.Generate(ctx, prompt, options)
}

// TestAJudgedAnswerIsReplayedForTheSameInputs holds a judged suite to asking the judge once: a judged question costs 6
// to 27 s of the card, and a reference's pictures are judged again for every candidate compared with it. Judged for
// another model that drew the same pictures, the suite puts no question to the judge and
// scores the same; a picture that changed is asked about again, and only it; and a judge with another identity is asked
// everything.
func TestAJudgedAnswerIsReplayedForTheSameInputs(t *testing.T) {
	t.Parallel()
	items := []TIFAItem{
		{ID: "fox", Caption: "a red fox", Questions: []TIFAQuestion{
			{Question: "is there a fox?", Choices: []string{"yes", "no"}, Answer: "yes"},
			{Question: "is the fox red?", Choices: []string{"yes", "no"}, Answer: "yes"}}},
		{ID: "boat", Caption: "a boat at sunrise", Questions: []TIFAQuestion{
			{Question: "is there a boat?", Choices: []string{"yes", "no"}, Answer: "yes"},
			{Question: "is it sunrise?", Choices: []string{"yes", "no"}, Answer: "yes"}}},
	}
	store := overgodbtest.Open(t, t.TempDir())
	chances := map[string]float64{"is there a fox?": 0.9, "is the fox red?": 0.8, "is there a boat?": 0.6, "is it sunrise?": 0.1}
	// judged evaluates the suite of pictures for the model named, by the judge named, and says how many questions the
	// judge was put, with the report.
	judged := func(model, judge string, images map[string]string) (int, generatedAnswerReport) {
		t.Helper()
		suite := testcheck.Must(BuildTIFAQuestions(items, images, judge))(t)
		compiled := testcheck.Must(CompileGeneratedAnswer(suite))(t)
		plan := testcheck.Must(bindGeneratedAnswer(compiled, ExactAuthorities{
			ModelDefinition: planID(t, artifact.KindModelDefinition, model), RuntimeRecipe: planID(t, artifact.KindRecipe, model),
			CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
			Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
		}))(t)
		publishPlanFixtureAuthorities(t, store, plan)
		asked := 0
		report := testcheck.Must(evaluateGeneratedAnswer(t.Context(), store, countedJudge{choiceJudge{judge, chances}, &asked}, compiled, plan))(t)
		return asked, report
	}
	pictures := map[string]string{"fox": strings.Repeat("a", 64) + imageExtension, "boat": strings.Repeat("b", 64) + imageExtension}
	asked, first := judged("reference", "judge-a", pictures)
	if asked != 4 {
		t.Fatalf("the first judgement put %d questions to the judge, want all 4", asked)
	}
	if asked, again := judged("candidate", "judge-a", pictures); asked != 0 || again.Accuracy != first.Accuracy || again.Soft != first.Soft {
		t.Fatalf("judged for another model that drew the same pictures, the suite put %d questions and scored %v and %v; want none, and %v and %v", asked, again.Accuracy, again.Soft, first.Accuracy, first.Soft)
	}
	changed := map[string]string{"fox": strings.Repeat("c", 64) + imageExtension, "boat": pictures["boat"]}
	if asked, _ := judged("redrawn", "judge-a", changed); asked != 2 {
		t.Fatalf("a changed fox picture put %d questions to the judge, want its 2", asked)
	}
	if asked, _ := judged("rejudged", "judge-b", pictures); asked != 4 {
		t.Fatalf("another judge was put %d questions, want all 4", asked)
	}
}
