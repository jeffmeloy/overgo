package evaluation

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// lmEvalRevision is the revision of TIGER-Lab/MMLU-Pro lm-eval reads, the latest in its cache, whose test split
// lm-eval's reference holds in order (mmlu_pro_reference_contents.py binds each question's content to it).
const lmEvalRevision = "dd36ce4b34827164989f100331f82c5a29741747"

// revisedQuestions are the test questions whose text differs between the revision lm-eval reads and the one overgo
// imports (9126953e, the lexically first), each with its content in both: set aside from the comparison, since the two
// are not asked the same question, and admitted as exactly that difference alone.
var revisedQuestions = map[int]struct{ lmEval, overgo string }{
	6065:  {lmEval: "97a54d966beb91ffe5f67a863dc76ff1e8e2d1b0a3d0a41b19edf1b7f8b1af33", overgo: "d391f4d21f84f52889f8641ffbbdbb4258b71fbf86e8bb7de2535121a2f35237"},
	11783: {lmEval: "e93445ce196c29f68a08d4ac941d2b565d0fd457c527ed075c0cf097a6fc24be", overgo: "8ab289e88312b132d1d6a4c3bf6950ed56c45816cade5848b02d2409d59454a1"},
}

// lmEvalReference is what mmlu_pro_lm_eval_reference.py keeps of lm-eval's reading, and
// mmlu_pro_reference_contents.py adds: its task's settings, the dataset revision it read, and each question's id,
// category, answer, content, the letter chosen and whether it was right, in the order of the test split.
type lmEvalReference struct {
	Task            string           `json:"task"`
	Model           string           `json:"model"`
	Dtype           string           `json:"dtype"`
	NumFewshot      int              `json:"num_fewshot"`
	FewshotSplit    string           `json:"fewshot_split"`
	TestSplit       string           `json:"test_split"`
	DatasetRevision string           `json:"dataset_revision"`
	Accuracy        float64          `json:"accuracy"`
	Questions       []lmEvalQuestion `json:"questions"`
}

// lmEvalQuestion is one question of lm-eval's reading: its id, category, answer and content, the letter lm-eval
// chose and whether it was right.
type lmEvalQuestion struct {
	QuestionID  int    `json:"question_id"`
	Category    string `json:"category"`
	AnswerIndex int    `json:"answer_index"`
	Content     string `json:"content"`
	Chosen      int    `json:"chosen"`
	Right       bool   `json:"right"`
}

// mmluProContent is a question's content as the reference records it: the sha256 of its question and options joined
// by the unit separator, as UTF-8.
func mmluProContent(question string, options []string) string {
	sum := sha256.Sum256([]byte(strings.Join(append([]string{question}, options...), "\x1f")))
	return hex.EncodeToString(sum[:])
}

// admitLmEvalReference admits lm-eval's reference against the MMLU-Pro test questions overgo imported, in their
// order, before any model opens: the revision lm-eval read, as many questions, each id once, and at each place the
// same question and options, category and answer, or a revised question's exact two versions (SQA-369).
func admitLmEvalReference(reference lmEvalReference, questions []storeCase) error {
	if reference.DatasetRevision != lmEvalRevision {
		return fmt.Errorf("the reference read MMLU-Pro revision %q, want %s", reference.DatasetRevision, lmEvalRevision)
	}
	if len(reference.Questions) != len(questions) {
		return fmt.Errorf("the reference holds %d questions, overgo's import %d", len(reference.Questions), len(questions))
	}
	held := map[int]bool{}
	for ordinal, entry := range questions {
		lmEval := reference.Questions[ordinal]
		if held[lmEval.QuestionID] || entry.ordinal != ordinal {
			return fmt.Errorf("place %d holds lm-eval's question %d, held before, or overgo's case %s/%d", ordinal, lmEval.QuestionID, entry.entry, entry.ordinal)
		}
		held[lmEval.QuestionID] = true
		question, options, answer, category, err := mmluProRow(entry)
		if err != nil {
			return err
		}
		content := mmluProContent(question, options)
		if revised, ok := revisedQuestions[lmEval.QuestionID]; ok {
			if lmEval.Content != revised.lmEval || content != revised.overgo {
				return fmt.Errorf("revised question %d reads %s in lm-eval and %s in overgo, want %s and %s", lmEval.QuestionID, lmEval.Content, content, revised.lmEval, revised.overgo)
			}
			continue
		}
		if lmEval.Content != content || lmEval.AnswerIndex != answer || lmEval.Category != category {
			return fmt.Errorf("place %d: lm-eval's question %d of %s answers %d, content %s; overgo's case %s/%d of %s answers %d, content %s",
				ordinal, lmEval.QuestionID, lmEval.Category, lmEval.AnswerIndex, lmEval.Content, entry.entry, entry.ordinal, category, answer, content)
		}
	}
	return nil
}

// TestTheMMLUProReferenceIsAdmittedByItsQuestions holds the comparison with lm-eval to the questions it compares
// (mmlu-pro/reference, SQA-369): it matched its reference by ordinal and count alone, so a reference of other
// questions at the same places passed until the answers were compared, after the whole suite had run. Before any
// model opens, a reference of the revision lm-eval read, with each question's content at its place, is admitted;
// another question at a place, another revision, a question fewer, an id twice, or a revised question not in its two
// exact versions is refused.
func TestTheMMLUProReferenceIsAdmittedByItsQuestions(t *testing.T) {
	t.Parallel()
	questions := mmluProFixture(t)[:4]
	reference := func() lmEvalReference {
		read := lmEvalReference{DatasetRevision: lmEvalRevision}
		for index, entry := range questions {
			question, options, answer, category, err := mmluProRow(entry)
			testcheck.NoError(t, err)
			read.Questions = append(read.Questions, lmEvalQuestion{QuestionID: index + 1, Category: category, AnswerIndex: answer, Content: mmluProContent(question, options)})
		}
		return read
	}
	if err := admitLmEvalReference(reference(), questions); err != nil {
		t.Fatalf("the reference of the imported questions was refused: %v", err)
	}
	for name, change := range map[string]func(*lmEvalReference){
		"another question at a place": func(read *lmEvalReference) {
			read.Questions[2].Content = mmluProContent("Which is even?", []string{"five", "four"})
		},
		"another revision":           func(read *lmEvalReference) { read.DatasetRevision = "9126953e3e196ff1073499437322f9e2c51ed79a" },
		"a question fewer":           func(read *lmEvalReference) { read.Questions = read.Questions[:3] },
		"an id twice":                func(read *lmEvalReference) { read.Questions[3].QuestionID = read.Questions[0].QuestionID },
		"a revised question changed": func(read *lmEvalReference) { read.Questions[1].QuestionID = 6065 },
	} {
		read := reference()
		change(&read)
		if err := admitLmEvalReference(read, questions); err == nil {
			t.Errorf("a reference with %s was admitted", name)
		}
	}
}
