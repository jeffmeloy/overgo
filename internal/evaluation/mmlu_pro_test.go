package evaluation

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/testutil"
)

// mmluProRequestsScript writes the fixture TestMMLUProAsksAsLMEvalDoes reads,
// from lm-eval's own task. It is kept where Python is kept, on the machine
// that ran it and not in git: the fixture is the data.
const mmluProRequestsScript = "scripts/python_golden_references/lm_eval_mmlu_pro_requests.py"

// recordingScorer scores every candidate alike and keeps what it was asked.
type recordingScorer struct {
	scores     []sequencescore.Score
	prompts    *[]string
	candidates *[][]string
}

func (s recordingScorer) ScoreContinuations(_ context.Context, prompt string, candidates []string) ([]sequencescore.Score, error) {
	*s.prompts = append(*s.prompts, prompt)
	*s.candidates = append(*s.candidates, slices.Clone(candidates))
	return s.scores, nil
}

// TestMMLUProAsksAsLMEvalDoes holds what a likelihood scoring of MMLU-Pro
// gives a model to the task the suite is a port of, lm-eval's
// leaderboard_mmlu_pro, as lm-eval itself builds its requests. The task
// writes each choice as a bare letter and leaves the space before it to its
// target delimiter, so what is scored after "Answer:" is " A" and a
// demonstration ends "Answer: A". Scored glued, as overgo scored it until
// this test, a letter is whatever token the tokenizer has for a colon and a
// letter together: one rare token for most letters and two for some, so that
// a small model's choice followed its tokenizer and not the question.
//
// For documents taken through the test split, the prompt and the candidates
// a likelihood scoring is handed must be lm-eval's context and continuations,
// with no demonstrations, as the store's suite is asked, and with the task's
// own five.
func TestMMLUProAsksAsLMEvalDoes(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile(filepath.Join(testutil.RepoRoot(t), "fixtures", "lm_eval", "leaderboard_mmlu_pro_requests.json"))
	if err != nil {
		t.Fatalf("%v; write the fixture with %s", err, mmluProRequestsScript)
	}
	type document struct {
		Question string   `json:"question"`
		Options  []string `json:"options"`
		Answer   int      `json:"answer"`
		Category string   `json:"category"`
	}
	type requests []struct {
		Document      int      `json:"document"`
		Context       string   `json:"context"`
		Continuations []string `json:"continuations"`
	}
	var reference struct {
		TargetDelimiter string              `json:"target_delimiter"`
		Documents       map[string]document `json:"documents"`
		Demonstrations  []document          `json:"demonstrations"`
		Without         requests            `json:"without_demonstrations"`
		WithFive        requests            `json:"with_five_demonstrations"`
	}
	if err := json.Unmarshal(raw, &reference); err != nil || len(reference.Without) == 0 || len(reference.WithFive) != len(reference.Without) {
		t.Fatalf("the fixture holds %d and %d requests: %v", len(reference.Without), len(reference.WithFive), err)
	}
	if reference.TargetDelimiter != targetDelimiter {
		t.Fatalf("lm-eval's target delimiter is %q and overgo's %q", reference.TargetDelimiter, targetDelimiter)
	}
	held := func(name string, source document) mmluProCase {
		return mmluProCase{Name: name, Category: source.Category, Question: source.Question, Options: source.Options, Answer: source.Answer}
	}
	for shots, asked := range map[int]requests{0: reference.Without, len(reference.Demonstrations): reference.WithFive} {
		suite := mmluProSuite{
			Kind: mmluProKind, Schema: "lm-eval/leaderboard-mmlu-pro/v0.1", Source: "fixture",
			Labels: choiceLetters, FewShotCount: shots, Normalization: sequencescore.NormalizationSum,
		}
		for index, demonstration := range reference.Demonstrations {
			suite.Demonstrations = append(suite.Demonstrations, held(fmt.Sprintf("validation/%d", index), demonstration))
		}
		for _, request := range asked {
			source, found := reference.Documents[fmt.Sprint(request.Document)]
			if !found {
				t.Fatalf("the fixture asks about document %d and does not hold it", request.Document)
			}
			suite.Cases = append(suite.Cases, held(fmt.Sprintf("test/%d", request.Document), source))
		}
		compiled, err := compileMMLUPro(suite)
		if err != nil {
			t.Fatal(err)
		}
		for index, scored := range compiled.choice.delimited().Cases {
			if scored.Prompt != asked[index].Context || !slices.Equal(scored.Candidates, asked[index].Continuations) {
				t.Errorf("with %d demonstrations, %s is asked as\n%q\nwith candidates %q, where lm-eval asks\n%q\nwith %q",
					shots, scored.Name, scored.Prompt, scored.Candidates, asked[index].Context, asked[index].Continuations)
			}
		}
	}
}

// TestMMLUProScoresCandidatesAfterTheDelimiter holds the scoring itself: the
// scorer is handed each candidate with the delimiter before it, the report
// groups accuracy by category, and the delimiter is no part of the suite. A
// likelihood plan that scores with it is another plan than one that glues the
// letters, so the records made before it are not taken for records of it,
// while the plan that asks the same questions through a chat template, which
// reads a letter whatever stands before it, is the plan it was, so that its
// records are still carried.
func TestMMLUProScoresCandidatesAfterTheDelimiter(t *testing.T) {
	t.Parallel()
	compiled, err := compileMMLUPro(mmluProSuite{
		Kind: mmluProKind, Schema: "lm-eval/leaderboard-mmlu-pro/v0.1", Source: "fixture",
		Labels: []string{"A", "B"}, FewShotCount: 1, Normalization: sequencescore.NormalizationSum,
		Demonstrations: []mmluProCase{
			{Name: "demo-1", Category: "math", Question: "Which number is even?", Options: []string{"three", "four"}, Answer: 1},
			{Name: "demo-2", Category: "math", Question: "unused", Options: []string{"x", "y"}, Answer: 0},
		},
		Cases: []mmluProCase{
			{Name: "color", Category: "knowledge", Question: "Which color is primary?", Options: []string{"red", "green"}, Answer: 0},
			{Name: "sum", Category: "math", Question: "One plus one?", Options: []string{"one", "two"}, Answer: 1},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := compiled.choice.suite.Cases[0].Candidates; !slices.Equal(got, []string{"A", "B"}) {
		t.Fatalf("the suite's own candidates are %q, want the bare labels", got)
	}
	authorities := func(asked prompting) ExactAuthorities {
		return ExactAuthorities{
			ModelDefinition: planID(t, artifact.KindModelDefinition, "model"),
			RuntimeRecipe:   planID(t, artifact.KindRecipe, "recipe"), CodeCommit: planTestCommit,
			Environment: planID(t, artifact.KindEvidence, "environment"), Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: asked},
		}
	}
	plan, err := bindMMLUPro(compiled, authorities(PromptingRawCompletion))
	if err != nil {
		t.Fatal(err)
	}
	glued := compiled
	glued.choice.delimiter = ""
	for asked, same := range map[prompting]bool{PromptingRawCompletion: false, PromptingChatTemplate: true} {
		with, withErr := bindMMLUPro(compiled, authorities(asked))
		without, withoutErr := bindMMLUPro(glued, authorities(asked))
		if withErr != nil || withoutErr != nil || (with.identity == without.identity) != same {
			t.Errorf("under %s the plan with the delimiter and the plan without it are the same plan: %v, want %v (%v, %v)",
				asked.Label(), with.identity == without.identity, same, withErr, withoutErr)
		}
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	var prompts []string
	var candidates [][]string
	scorer := recordingScorer{
		scores: []sequencescore.Score{{LogProbability: -1, Tokens: 1}, {LogProbability: -2, Tokens: 1}}, prompts: &prompts, candidates: &candidates,
	}
	report, err := evaluateMMLUPro(t.Context(), store, scorer, compiled, plan)
	if err != nil {
		t.Fatal(err)
	}
	const first = "Which number is even?\nA. three\nB. four\nAnswer: B\n\nWhich color is primary?\nA. red\nB. green\nAnswer:"
	if len(prompts) != 2 || prompts[0] != first || !slices.Equal(candidates[0], []string{" A", " B"}) || !slices.Equal(candidates[1], []string{" A", " B"}) {
		t.Fatalf("the scorer was asked %q with %q", prompts, candidates)
	}
	if report.Accuracy != 0.5 || len(report.Categories) != 2 || report.Categories[0].Name != "knowledge" ||
		report.Categories[0].Accuracy != 1 || report.Categories[1].Name != "math" || report.Categories[1].Accuracy != 0 {
		t.Fatalf("MMLU-Pro report = %+v", report)
	}
}
