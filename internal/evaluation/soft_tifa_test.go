package evaluation

import (
	"context"
	"math"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// choiceJudge answers each question of a picture with the letter its logits favour, the logits giving the first
// letter the probability rightChance names by question.
type choiceJudge struct {
	judge       string
	rightChance map[string]float64
}

func (g choiceJudge) Judge() string { return g.judge }

// ChoiceTokens names letter A token 0 and letter B token 1.
func (choiceJudge) ChoiceTokens(choices []string) ([]tokenizer.TokenID, error) {
	return []tokenizer.TokenID{0, 1}[:len(choices)], nil
}

func (g choiceJudge) Generate(_ context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: exactPromptTokens})
	for question, chance := range g.rightChance {
		if strings.Contains(prompt, question) {
			// The chance is the first choice's, letter A, the right answer of every question the fixture does not correct.
			logits, piece := []float32{float32(math.Log(chance)), float32(math.Log(1 - chance))}, "A"
			if chance < 0.5 {
				piece = "B"
			}
			return make([]tokenizer.TokenID, exactPromptTokens+1), "", options.OnToken(inference.TokenEvent{Piece: piece, Logits: logits})
		}
	}
	return nil, "", nil
}

// tifaItems are the judged-picture fixture: two captions' questions, each answered "yes", its first choice, and
// tifaChances the judge's probability of each question's first choice, one of them a confident wrong answer.
var (
	tifaItems = []TIFAItem{
		{ID: "fox", Caption: "a red fox", Questions: []TIFAQuestion{
			{Question: "is there a fox?", Choices: []string{"yes", "no"}, Answer: "yes"},
			{Question: "is the fox red?", Choices: []string{"yes", "no"}, Answer: "yes"}}},
		{ID: "boat", Caption: "a boat at sunrise", Questions: []TIFAQuestion{
			{Question: "is there a boat?", Choices: []string{"yes", "no"}, Answer: "yes"},
			{Question: "is it sunrise?", Choices: []string{"yes", "no"}, Answer: "yes"}}},
	}
	tifaChances  = map[string]float64{"is there a fox?": 0.9, "is the fox red?": 0.8, "is there a boat?": 0.6, "is it sunrise?": 0.1}
	tifaPictures = map[string]string{"fox": strings.Repeat("a", 64) + imageExtension, "boat": strings.Repeat("b", 64) + imageExtension}
)

// tifaPlan compiles items' questions about images, judged by judge, for the model named, and binds and publishes the
// plan in store.
func tifaPlan(t *testing.T, store *overgodb.Store, model, judge string, items []TIFAItem, images map[string]string) (generatedAnswerPlan, Plan) {
	t.Helper()
	suite := testcheck.Must(BuildTIFAQuestions(items, images, judge))(t)
	compiled := testcheck.Must(CompileGeneratedAnswer(suite))(t)
	plan := testcheck.Must(bindGeneratedAnswer(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, model), RuntimeRecipe: planID(t, artifact.KindRecipe, model),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	}))(t)
	publishPlanFixtureAuthorities(t, store, plan)
	return compiled, plan
}

// TestSoftTIFAScoresTheProbabilityOfTheRightAnswer holds TIFA's picture questions to how sure the judge is: on
// 2026-11-08 binary TIFA read 0.94 for Krea-2-Turbo and 100 of 100 for SenseNova, at the ceiling, where a worse picture
// cannot lower it. Each question is a choice among its letters, the judge's probability of the right letter is read
// from its first token's logits, and a caption scores the geometric mean over its questions: a confident wrong answer
// lowers the soft share below the binary one. A runtime that names no token of the choices is refused.
func TestSoftTIFAScoresTheProbabilityOfTheRightAnswer(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	compiled, plan := tifaPlan(t, store, "model", "judge-a", tifaItems, tifaPictures)
	report := testcheck.Must(evaluateGeneratedAnswer(t.Context(), store, choiceJudge{"judge-a", tifaChances}, compiled, plan))(t)
	want := (math.Sqrt(0.9*0.8) + math.Sqrt(0.6*0.1)) / 2
	if report.Accuracy != 0.75 || math.Abs(report.Soft-want) > 1e-5 || report.Soft >= report.Accuracy {
		t.Fatalf("the report reads accuracy %v and soft %v, want 0.75 and %.4f, the confident wrong answer below the binary share", report.Accuracy, report.Soft, want)
	}
	metrics := generatedAnswerMetrics(compiled.suite, report.Accuracy, report.Soft)
	if len(metrics) != 2 || metrics[1].Name != SoftMetric || metrics[1].Value != report.Soft {
		t.Fatalf("the suite's metrics are %+v, want the accuracy and the soft share", metrics)
	}
	// A store that keeps none of the judge's answers, which would otherwise be read rather than asked (replayJudged).
	fresh := overgodbtest.Open(t, t.TempDir())
	publishPlanFixtureAuthorities(t, fresh, plan)
	if _, err := evaluateGeneratedAnswer(t.Context(), fresh, judgedGenerator{exactGenerator{pieces: []string{"A"}}, "judge-a"}, compiled, plan); err == nil {
		t.Fatal("a judge that names no token of the choices scored a suite of choices")
	}
}
