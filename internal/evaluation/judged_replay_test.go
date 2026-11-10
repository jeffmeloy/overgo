package evaluation

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/overgodb"
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

// judgeTIFA evaluates items' questions about images in store for the model named, by the judge named, and says how many
// questions the judge was put, with the report.
func judgeTIFA(t *testing.T, store *overgodb.Store, model, judge string, items []TIFAItem, images map[string]string) (int, generatedAnswerReport) {
	t.Helper()
	compiled, plan := tifaPlan(t, store, model, judge, items, images)
	asked := 0
	report := testcheck.Must(evaluateGeneratedAnswer(t.Context(), store, countedJudge{choiceJudge{judge, tifaChances}, &asked}, compiled, plan))(t)
	return asked, report
}

// TestAJudgedAnswerIsReplayedForTheSameInputs holds a judged suite to asking the judge once: a judged question costs 6
// to 27 s of the card, and a reference's pictures are judged again for every candidate compared with it. Judged for
// another model that drew the same pictures, the suite puts no question to the judge and
// scores the same; a picture that changed is asked about again, and only it; and a judge with another identity is asked
// everything.
func TestAJudgedAnswerIsReplayedForTheSameInputs(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	asked, first := judgeTIFA(t, store, "reference", "judge-a", tifaItems, tifaPictures)
	if asked != 4 {
		t.Fatalf("the first judgement put %d questions to the judge, want all 4", asked)
	}
	if asked, again := judgeTIFA(t, store, "candidate", "judge-a", tifaItems, tifaPictures); asked != 0 || again.Accuracy != first.Accuracy || again.Soft != first.Soft {
		t.Fatalf("judged for another model that drew the same pictures, the suite put %d questions and scored %v and %v; want none, and %v and %v", asked, again.Accuracy, again.Soft, first.Accuracy, first.Soft)
	}
	changed := map[string]string{"fox": strings.Repeat("c", 64) + imageExtension, "boat": tifaPictures["boat"]}
	if asked, _ := judgeTIFA(t, store, "redrawn", "judge-a", tifaItems, changed); asked != 2 {
		t.Fatalf("a changed fox picture put %d questions to the judge, want its 2", asked)
	}
	if asked, _ := judgeTIFA(t, store, "rejudged", "judge-b", tifaItems, tifaPictures); asked != 4 {
		t.Fatalf("another judge was put %d questions, want all 4", asked)
	}
}

// TestACorrectedAnswerIsNotScoredByTheOldAnswersProbability judges the fixture's pictures once, then again after one
// question's answer is corrected from its first choice to its second (SQA-343). The judge kept the probability of the
// first choice, 0.1 for "is it sunrise?", and the corrected right answer has 0.9: the corrected question is put to the
// judge again and scored by the new answer's probability, while the three unchanged ones are read as they were kept.
func TestACorrectedAnswerIsNotScoredByTheOldAnswersProbability(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	if asked, _ := judgeTIFA(t, store, "model", "judge-a", tifaItems, tifaPictures); asked != 4 {
		t.Fatalf("the first judgement put %d questions to the judge, want all 4", asked)
	}
	corrected := slices.Clone(tifaItems)
	corrected[1].Questions = slices.Clone(tifaItems[1].Questions)
	corrected[1].Questions[1].Answer = "no"
	asked, report := judgeTIFA(t, store, "corrected", "judge-a", corrected, tifaPictures)
	want := (math.Sqrt(0.9*0.8) + math.Sqrt(0.6*0.9)) / 2
	if asked != 1 || report.Accuracy != 1 || math.Abs(report.Soft-want) > 1e-5 {
		t.Fatalf("after the correction the judge was put %d questions and the suite reads accuracy %v and soft %v; want the corrected question alone asked again, every answer right and %.4f", asked, report.Accuracy, report.Soft, want)
	}
}
