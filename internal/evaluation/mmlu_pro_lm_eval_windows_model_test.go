//go:build windows && modeltest

package evaluation

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// lmEvalRoot is the lm-eval install overgo's leaderboard reading of MMLU-Pro is held to: its dataset cache, which
// overgo's MMLU-Pro import reads, and the reference lm-eval's own reading of Qwen2.5-0.5B left there
// (scripts/python_golden_references/mmlu_pro_lm_eval_reference.py).
const lmEvalRoot = "C:/Users/jeffm/eval"

// The bar mmlu-pro-lm-eval/matches-lm-eval declared before either reading: over the questions both read alike,
// overgo's accuracy within a point of lm-eval's, and the same letter chosen on at least nine questions in ten.
const (
	lmEvalAccuracyBar  = 0.01
	lmEvalAgreementBar = 0.9
)

// revisedQuestions are the test questions whose text differs between the dataset revision lm-eval reads (dd36ce4b,
// the latest cached) and the one overgo imports (9126953e, the lexically first): set aside, since the two are not
// asked the same question.
var revisedQuestions = []int{6065, 11783}

// lmEvalReference is what mmlu_pro_lm_eval_reference.py keeps of lm-eval's reading: its task's settings, and each
// question's id, category, answer, the letter chosen and whether it was right, in the order of the test split.
type lmEvalReference struct {
	Task         string  `json:"task"`
	Model        string  `json:"model"`
	Dtype        string  `json:"dtype"`
	NumFewshot   int     `json:"num_fewshot"`
	FewshotSplit string  `json:"fewshot_split"`
	TestSplit    string  `json:"test_split"`
	Accuracy     float64 `json:"accuracy"`
	Questions    []struct {
		QuestionID  int    `json:"question_id"`
		Category    string `json:"category"`
		AnswerIndex int    `json:"answer_index"`
		Chosen      int    `json:"chosen"`
		Right       bool   `json:"right"`
	} `json:"questions"`
}

// TestMMLUProReadsAsLmEvalReadsIt reads Qwen2.5-0.5B, a model lm-eval in lmEvalRoot loads too, by overgo's leaderboard
// reading of MMLU-Pro (five validation demonstrations, plain text, each letter after the target delimiter) over the
// whole test split, imported from the cache lm-eval reads, and holds it to lm-eval 0.4.9.1's own reading of the model
// by leaderboard_mmlu_pro within the bar declared before either reading: questions matched by their place in the split
// and their question_id, the revised ones set aside, accuracy within lmEvalAccuracyBar and the same letter chosen on at
// least lmEvalAgreementBar of them. lm-eval reads the model in float32: read in bfloat16 (its config's), 1,547 of the
// questions had its top two letters round to a tie, which its argmax broke to the first (the row's note).
func TestMMLUProReadsAsLmEvalReadsIt(t *testing.T) {
	// Serial: the model is loaded alone on the card.
	cudatest.Require(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	var reference lmEvalReference
	data := testcheck.Must(os.ReadFile(filepath.Join(lmEvalRoot, "overgo_qwen25_mmlu_pro_float32_results", "reference.json")))(t)
	testcheck.NoError(t, json.Unmarshal(data, &reference))
	if reference.Task != "leaderboard_mmlu_pro" || reference.NumFewshot != leaderboardShots || reference.FewshotSplit != "validation" ||
		reference.TestSplit != "test" || reference.Model != "Qwen2.5-0.5B" || reference.Dtype != "torch.float32" {
		t.Fatalf("the reference is %s of %s in %s, %d shots from %s, read on %s; want leaderboard_mmlu_pro of Qwen2.5-0.5B in float32, %d from validation, on test",
			reference.Task, reference.Model, reference.Dtype, reference.NumFewshot, reference.FewshotSplit, reference.TestSplit, leaderboardShots)
	}

	// The MMLU-Pro splits of the cache lm-eval reads, imported as the import reads them, into a store of the test's own.
	var declarations []benchmarkDeclaration
	for _, seed := range testcheck.Must(deriveHFCacheBenchmarks(filepath.Join(lmEvalRoot, "datasets")))(t) {
		if strings.HasPrefix(seed.Name, "mmlu-pro/") {
			declarations = append(declarations, benchmarkDeclaration{Name: seed.Name, Path: seed.Path, Spec: seed.Spec})
		}
	}
	answers := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	defer answers.Close()
	testcheck.Must(catalogBenchmarkDeclarations(t.Context(), answers, "", declarations))(t)
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "qwen2.5-0.5b"), RuntimeRecipe: planID(t, artifact.KindRecipe, "qwen2.5-0.5b"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "device"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingRawCompletion},
	}
	suites, _, err := DeriveStoreSuiteFamily(t.Context(), answers, authorities, "mmlu-pro"+wholeFamily)
	if err != nil || len(suites) != 1 || int(suites[0].Descriptor().Cases) != len(reference.Questions) {
		t.Fatalf("the cache derives %d MMLU-Pro suites, the first of %s cases, where lm-eval read %d: %v", len(suites), cases(suites), len(reference.Questions), err)
	}
	suite := suites[0]

	runner := testcheck.Must(modelcli.OpenRunner(t.Context(), roots.Store, filepath.Join(roots.Models, "Qwen2.5-0.5B"), inference.OpenOptions{}))(t)
	defer func() { testcheck.NoError(t, runner.Close()) }()
	publishPlanFixtureAuthorities(t, answers, suite.plan)
	result := testcheck.Must(suite.execute(context.WithoutCancel(t.Context()), answers, runner))(t)
	content := testcheck.Must(artifact.RequireTypedContent(t.Context(), answers, result.Report))(t)
	var report struct {
		Observations []struct {
			Name     string
			Selected *int
			Answer   *int
		}
	}
	testcheck.NoError(t, json.Unmarshal(content.Data, &report))

	type tally struct{ asked, overgo, lmEval int }
	categories := map[string]*tally{}
	var asked, agreed, overgoRight, lmEvalRight, onlyOvergo, onlyLmEval int
	for _, observation := range report.Observations {
		ordinal := testcheck.Must(strconv.Atoi(observation.Name[strings.LastIndex(observation.Name, "/")+1:]))(t)
		question := reference.Questions[ordinal]
		if observation.Selected == nil || observation.Answer == nil || *observation.Answer != question.AnswerIndex {
			t.Fatalf("case %s answered %v of answer %v, where lm-eval's question %d answers %d", observation.Name, observation.Selected, observation.Answer, question.QuestionID, question.AnswerIndex)
		}
		if slices.Contains(revisedQuestions, question.QuestionID) {
			continue
		}
		right := *observation.Selected == question.AnswerIndex
		asked++
		if *observation.Selected == question.Chosen {
			agreed++
		}
		counted := categories[question.Category]
		if counted == nil {
			counted = &tally{}
			categories[question.Category] = counted
		}
		counted.asked++
		switch {
		case right && question.Right:
			overgoRight, lmEvalRight = overgoRight+1, lmEvalRight+1
			counted.overgo, counted.lmEval = counted.overgo+1, counted.lmEval+1
		case right:
			overgoRight, onlyOvergo, counted.overgo = overgoRight+1, onlyOvergo+1, counted.overgo+1
		case question.Right:
			lmEvalRight, onlyLmEval, counted.lmEval = lmEvalRight+1, onlyLmEval+1, counted.lmEval+1
		}
	}
	if asked != len(reference.Questions)-len(revisedQuestions) {
		t.Fatalf("%d questions compared, want the %d lm-eval read less the %d revised", asked, len(reference.Questions), len(revisedQuestions))
	}
	overgoAccuracy, lmEvalAccuracy := float64(overgoRight)/float64(asked), float64(lmEvalRight)/float64(asked)
	agreement := float64(agreed) / float64(asked)
	t.Logf("over %d questions: overgo %.4f, lm-eval %.4f (%.4f over all %d), difference %+.4f; the same letter on %.4f; right in overgo alone %d, in lm-eval alone %d",
		asked, overgoAccuracy, lmEvalAccuracy, reference.Accuracy, len(reference.Questions), overgoAccuracy-lmEvalAccuracy, agreement, onlyOvergo, onlyLmEval)
	names := make([]string, 0, len(categories))
	for name := range categories {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		counted := categories[name]
		t.Logf("  %-16s %4d questions: overgo %.3f, lm-eval %.3f", name, counted.asked, float64(counted.overgo)/float64(counted.asked), float64(counted.lmEval)/float64(counted.asked))
	}
	if math.Abs(overgoAccuracy-lmEvalAccuracy) > lmEvalAccuracyBar || agreement < lmEvalAgreementBar {
		t.Errorf("overgo reads Qwen2.5-0.5B at %.4f against lm-eval's %.4f, choosing the same letter on %.4f of %d questions; the bar is within %.2f and at least %.2f",
			overgoAccuracy, lmEvalAccuracy, agreement, asked, lmEvalAccuracyBar, lmEvalAgreementBar)
	}
}

// cases is how many cases the first of suites holds, none where there is none.
func cases(suites []CompiledSuite) string {
	if len(suites) == 0 {
		return "no"
	}
	return fmt.Sprint(suites[0].Descriptor().Cases)
}
