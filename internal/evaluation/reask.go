package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// AnswerVariant is a way of asking a chat-template choice again (answer-format/reread): the words the model's turn
// opens with and how many tokens its answer is read from.
type AnswerVariant struct {
	Opener string
	Tokens int
}

// Reasked is what asking a reading's budget-ended answers again found: how many there were, and for how many the
// variant drew an option letter.
type Reasked struct {
	Asked, Lettered int
}

// PreparedReask is the answers of an evaluation record's chat-template reading that named no option because the token
// budget ended (UnreadCauses.Budget), made ready to be asked again: their cases, read from the record, its report and
// its dataset. A protocol chosen on what they answer makes them seen, so preparing them publishes them as development
// evidence of their dataset (exposeDevelopment) before any is asked, and a held-out part that holds one is refused a
// confirmation (SQA-361).
type PreparedReask struct {
	cases []MultipleChoiceCase
}

// PrepareReask admits record's budget-ended answers and publishes their cases as development evidence into store. A
// record, report or dataset that cannot be read, or a variant that cannot be asked, refuses before anything is
// published; a refusal to publish refuses before anything is asked. It opens no model.
func PrepareReask(ctx context.Context, store *overgodb.Store, record artifact.ID, variants []AnswerVariant) (PreparedReask, error) {
	if len(variants) == 0 {
		return PreparedReask{}, errors.New("evaluation: a re-ask names at least one variant")
	}
	for _, variant := range variants {
		if variant.Opener == "" || variant.Tokens <= 0 {
			return PreparedReask{}, errors.New("evaluation: a variant names its opener and a positive token budget")
		}
	}
	evaluated, err := runrecord.RequireEvaluation(ctx, store, record)
	if err != nil {
		return PreparedReask{}, err
	}
	bound, err := runrecord.RequireRun(ctx, store, evaluated.Run)
	if err != nil {
		return PreparedReask{}, err
	}
	_, report, read := scoredReport(ctx, store, bound)
	if !read {
		return PreparedReask{}, fmt.Errorf("evaluation: the report of %s cannot be read", record)
	}
	dataset, found, err := artifact.ReadContent(ctx, store, evaluated.Dataset)
	if err != nil || !found {
		return PreparedReask{}, errors.Join(err, fmt.Errorf("evaluation: the dataset of %s cannot be read", record))
	}
	var cases []MultipleChoiceCase
	if err := json.Unmarshal(dataset.Data, &cases); err != nil {
		return PreparedReask{}, fmt.Errorf("evaluation: the dataset of %s holds no choices: %w", record, err)
	}
	byName := make(map[string]MultipleChoiceCase, len(cases))
	for _, testCase := range cases {
		byName[testCase.Name] = testCase
	}
	var prepared PreparedReask
	for _, answer := range reportAnswers(report.Data) {
		var cause UnreadCauses
		if answer.Selected != unmatchedChoice {
			continue
		}
		if cause.add(answer.Raw, answer.Ended); cause.Budget == 0 {
			continue
		}
		testCase, held := byName[answer.Name]
		if !held {
			return PreparedReask{}, fmt.Errorf("evaluation: %s's dataset holds no case %s", record, answer.Name)
		}
		if _, _, err := chatChoicePrompt(testCase, chatChoiceInstruction); err != nil {
			return PreparedReask{}, err
		}
		prepared.cases = append(prepared.cases, testCase)
	}
	if len(prepared.cases) == 0 {
		return prepared, nil
	}
	return prepared, exposeDevelopment(ctx, store, evaluated.Dataset, prepared.cases)
}

// Ask puts each prepared answer to the model again under variant, through the same template and question, greedily,
// and counts those it now answers with an option letter, read as the protocol reads one (matchChoiceLetter). It
// records no answer: the measurement decides whether the protocol changes.
func (prepared PreparedReask) Ask(ctx context.Context, runtime chatChoiceRuntime, variant AnswerVariant) (Reasked, error) {
	var reasked Reasked
	for _, testCase := range prepared.cases {
		prompt, _, err := chatChoicePrompt(testCase, chatChoiceInstruction)
		if err != nil {
			return reasked, err
		}
		shaped, err := runtime.ShapeChatPrompt(prompt)
		if err != nil {
			return reasked, err
		}
		written, err := Record(ctx, runtime, testCase.Name, shaped+variant.Opener, variant.Tokens)
		if err != nil {
			return reasked, err
		}
		reasked.Asked++
		if _, lettered := matchChoiceLetter(written.Text, testCase.Candidates); lettered {
			reasked.Lettered++
		}
	}
	return reasked, nil
}
