package evaluation

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/overgodbtest"
	"overgo/internal/sequencescore"
	"overgo/internal/testcheck"
	"overgo/internal/tokenizer"
)

// askedContexts is a model that records each context a likelihood reading hands it, with the continuations it is
// asked to score after it, and likes the first continuation best.
type askedContexts struct {
	contexts      []string
	continuations [][]string
}

func (m *askedContexts) ScoreContinuations(_ context.Context, context string, continuations []string) ([]sequencescore.Score, error) {
	m.contexts, m.continuations = append(m.contexts, context), append(m.continuations, continuations)
	scores := make([]sequencescore.Score, len(continuations))
	for index := range scores {
		scores[index] = sequencescore.Score{LogProbability: -float64(index + 1), Tokens: 1}
	}
	return scores, nil
}

func (*askedContexts) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("a likelihood reading generates nothing")
}

// TestMMLUProIsReadAsLeaderboardMMLUProReadsIt reads MMLU-Pro from a catalog holding its test and validation splits
// by likelihood on the plain text (mmlu-pro-lm-eval/five-shot), and wants each question handed to the model as lm-eval
// 0.4.9.1's leaderboard_mmlu_pro builds its context: the first five validation rows (num_fewshot 5, fewshot_split
// validation, sampler first_n), each written by the task's doc_to_text, then the target delimiter and its answer's
// letter, joined and closed by the fewshot delimiter, then the question by doc_to_text; and the letters scored after
// the target delimiter. The strings are written here from the task's utils.py and lm-eval's defaults, not from
// overgo's own. Asked through the chat template, the suite is the one the test split alone derives.
func TestMMLUProIsReadAsLeaderboardMMLUProReadsIt(t *testing.T) {
	t.Parallel()
	family := hfCacheFamilies["TIGER-Lab___mmlu-pro"]
	type row struct {
		question string
		options  []string
		answer   int
	}
	test := []row{{"What is 2 + 2?", []string{"3", "4", "5"}, 1}, {"Which is a noun?\nPick one.", []string{"run", "dog", "blue", "slowly"}, 1}}
	validation := []row{
		{"Shot one?", []string{"a", "b"}, 0}, {"Shot two?", []string{"a", "b", "c"}, 2}, {"Shot three?", []string{"x", "y"}, 1},
		{"Shot four?", []string{"p", "q", "r", "s"}, 3}, {"Shot five?", []string{"m", "n"}, 0}, {"Shot six, never shown?", []string{"u", "v"}, 1},
	}
	root := t.TempDir()
	declare := func(split string, rows []row) benchmarkDeclaration {
		var data bytes.Buffer
		for _, r := range rows {
			options := `"` + strings.Join(r.options, `","`) + `"`
			fmt.Fprintf(&data, `{"question":%q,"options":[%s],"answer_index":%d,"category":"math"}`+"\n", r.question, options, r.answer)
		}
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, split+".jsonl"), data.Bytes(), 0o600))
		digest := sha256.Sum256(data.Bytes())
		return benchmarkDeclaration{Name: "mmlu-pro/default/" + split, Path: split + ".jsonl", Spec: dataset.BenchmarkImportSpec{
			Source: "mmlu-pro:default", Revision: "0.0.0/9126953e", SHA256: hex.EncodeToString(digest[:]), Split: split,
			Format: dataset.BenchmarkFormatJSONL, Conversion: family.conversion, Fields: family.fields,
		}}
	}
	both := []benchmarkDeclaration{declare("test", test), declare("validation", validation)}
	derive := func(prompting prompting, declarations []benchmarkDeclaration) (CompiledSuite, artifact.Repository) {
		authorities := ListingAuthorities()
		authorities.Execution.Prompting = prompting
		store := overgodbtest.Open(t, t.TempDir())
		testcheck.Must(catalogBenchmarkDeclarations(t.Context(), store, root, declarations))(t)
		suites, _, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, "mmlu-pro"+wholeFamily)
		if err != nil || len(suites) != 1 {
			t.Fatalf("the catalog derives %d MMLU-Pro suites under %s: %v", len(suites), prompting.Label(), err)
		}
		return suites[0], store
	}

	// lm-eval's leaderboard/mmlu_pro/utils.py doc_to_text, and its defaults: target_delimiter " ", fewshot_delimiter "\n\n".
	docToText := func(r row) string {
		var text strings.Builder
		text.WriteString(r.question + "\n")
		for index, option := range r.options {
			text.WriteString(string(rune('A'+index)) + ". " + option + "\n")
		}
		return text.String() + "Answer:"
	}
	var shots strings.Builder
	for _, shot := range validation[:5] {
		shots.WriteString(docToText(shot) + " " + string(rune('A'+shot.answer)) + "\n\n")
	}
	suite, store := derive(PromptingRawCompletion, both)
	publishPlanFixtureAuthorities(t, store, suite.Plan())
	model := &askedContexts{}
	testcheck.Must(executeSuite(t.Context(), store, model, suite))(t)
	if len(model.contexts) != len(test) {
		t.Fatalf("the reading asked the model %d contexts, want one for each of the %d questions", len(model.contexts), len(test))
	}
	for index, question := range test {
		if want := shots.String() + docToText(question); model.contexts[index] != want {
			t.Errorf("question %d was asked as\n%q\nwant leaderboard_mmlu_pro's context\n%q", index, model.contexts[index], want)
		}
		var letters []string
		for option := range question.options {
			letters = append(letters, " "+string(rune('A'+option)))
		}
		if !slices.Equal(model.continuations[index], letters) {
			t.Errorf("question %d scored %q, want each letter after the target delimiter, %q", index, model.continuations[index], letters)
		}
	}
	chat, _ := derive(PromptingChatTemplate, both)
	alone, _ := derive(PromptingChatTemplate, both[:1])
	if chat.Plan().Identity() != alone.Plan().Identity() {
		t.Errorf("asked through the chat template, the suite binds plan %s beside the validation split and %s without it; want it asked alone", chat.Plan().Identity(), alone.Plan().Identity())
	}
}
