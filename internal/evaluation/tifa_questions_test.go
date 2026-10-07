package evaluation

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// tifaFile writes a question-answer file of captions with their questions, and
// one row of each kind a reader must skip.
func tifaFile(t *testing.T, captions int) string {
	t.Helper()
	var rows []tifaRow
	for caption := range captions {
		id := fmt.Sprintf("coco_%d", caption)
		rows = append(rows,
			tifaRow{ID: id, Caption: "caption " + id, Question: "is this a dog?", Choices: []string{"yes", "no"}, Answer: "yes"},
			tifaRow{ID: id, Caption: "caption " + id, Question: "what colour is it?", Choices: []string{"red", "blue", "green", "black"}, Answer: "green"})
	}
	rows = append(rows,
		tifaRow{ID: "coco_bad", Caption: "bad", Question: "answer not offered?", Choices: []string{"yes", "no"}, Answer: "maybe"},
		tifaRow{ID: "coco_bad", Caption: "bad", Question: "one choice?", Choices: []string{"yes"}, Answer: "yes"})
	data, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "questions.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestTIFAQuestionsAskAboutThePictureOfEachCaption holds the suite to its
// construction: the captions are picked by seed from those with a well-formed
// question, each question is a case naming the picture by its digest and
// lettering its choices in order, and the right answer is the letter of the
// choice the file gives.
func TestTIFAQuestionsAskAboutThePictureOfEachCaption(t *testing.T) {
	t.Parallel()
	path := tifaFile(t, 8)
	items, err := PickTIFA(path, 3, 17)
	if err != nil {
		t.Fatal(err)
	}
	again, _ := PickTIFA(path, 3, 17)
	if len(items) != 3 || fmt.Sprint(items) != fmt.Sprint(again) {
		t.Fatalf("the same seed picked %v and then %v", items, again)
	}
	images := map[string]string{}
	for _, item := range items {
		if item.ID == "coco_bad" || len(item.Questions) != 2 {
			t.Fatalf("picked %+v, which is no well-formed caption of two questions", item)
		}
		images[item.ID] = strings.Repeat("a", 64) + imageExtension
	}
	suite, err := BuildTIFAQuestions(items, images, "judge-a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := CompileGeneratedAnswer(suite); err != nil || suite.Judge != "judge-a" || len(suite.Cases) != 6 {
		t.Fatalf("suite = %d cases judged by %q (%v), want 6 cases judged by judge-a that compile", len(suite.Cases), suite.Judge, err)
	}
	kind, name, question, ok := ParseMediaPrompt(suite.Cases[1].Prompt)
	if !ok || kind != MediaImage || name != strings.Repeat("a", 64)+imageExtension ||
		!strings.Contains(question, "A. red\nB. blue\nC. green\nD. black") || suite.Cases[1].Answers[0] != "C" || suite.Cases[0].Answers[0] != "A" {
		t.Fatalf("second case = %q answering %v", suite.Cases[1].Prompt, suite.Cases[1].Answers)
	}
	if _, err := BuildTIFAQuestions(items, map[string]string{}, "judge-a"); err == nil {
		t.Fatal("a suite was built for captions with no picture")
	}
	if _, err := BuildTIFAQuestions(items, images, ""); err == nil {
		t.Fatal("a suite was built with no judge named")
	}
	if _, err := PickTIFA(path, 99, 17); err == nil {
		t.Fatal("more captions than the file holds were picked")
	}
}

type judgedGenerator struct {
	exactGenerator
	judge string
}

func (g judgedGenerator) Judge() string { return g.judge }

// TestAJudgedSuiteIsAnsweredOnlyByItsJudge holds the generated-answer suite to
// the judge it names: a runtime answering as another judge, or as none, is
// refused, and one answering as the named judge is scored.
func TestAJudgedSuiteIsAnsweredOnlyByItsJudge(t *testing.T) {
	t.Parallel()
	suite := GeneratedAnswerSuite{
		Kind: GeneratedAnswerKind, Schema: "fixture/v1", Source: "fixture", Judge: "judge-a", Transforms: []string{TransformTrimSpace},
		Cases: []GeneratedAnswerCase{{Name: "case", Prompt: "prompt", MaxTokens: 1, Answers: []string{"A"}}},
	}
	compiled, err := CompileGeneratedAnswer(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindGeneratedAnswer(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	answers := exactGenerator{pieces: []string{"A"}}
	if _, err := evaluateGeneratedAnswer(t.Context(), store, answers, compiled, plan); err == nil {
		t.Fatal("a runtime that is no judge answered a judged suite")
	}
	if _, err := evaluateGeneratedAnswer(t.Context(), store, judgedGenerator{answers, "judge-b"}, compiled, plan); err == nil {
		t.Fatal("a runtime answering as another judge answered the suite")
	}
	report, err := evaluateGeneratedAnswer(t.Context(), store, judgedGenerator{answers, "judge-a"}, compiled, plan)
	if err != nil || report.Accuracy != 1 {
		t.Fatalf("the named judge's report = %+v (%v), want accuracy 1", report, err)
	}
}
