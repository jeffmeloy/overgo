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
// opens with and how many tokens its answer is read from. The protocol's own is chatAnswerOpener and
// chatAnswerTokens.
type AnswerVariant struct {
	Opener string
	Tokens int
}

// Reasked is what asking a reading's budget-ended answers again found: how many there were, and for how many the
// variant drew an option letter.
type Reasked struct {
	Asked, Lettered int
}

// ReaskBudgetEnded puts each answer of an evaluation record's chat-template reading that named no option because the
// token budget ended (UnreadCauses.Budget) to the model again under variant, through the same template and question,
// greedily, and counts those it now answers with an option letter, read as the protocol reads one
// (matchChoiceLetter). It records nothing: the measurement decides whether the protocol changes.
func ReaskBudgetEnded(ctx context.Context, store *overgodb.Store, record artifact.ID, runtime chatChoiceRuntime, variant AnswerVariant) (Reasked, error) {
	if variant.Opener == "" || variant.Tokens <= 0 {
		return Reasked{}, errors.New("evaluation: a variant names its opener and a positive token budget")
	}
	evaluated, err := runrecord.RequireEvaluation(ctx, store, record)
	if err != nil {
		return Reasked{}, err
	}
	bound, err := runrecord.RequireRun(ctx, store, evaluated.Run)
	if err != nil {
		return Reasked{}, err
	}
	_, report, read := scoredReport(ctx, store, bound)
	if !read {
		return Reasked{}, fmt.Errorf("evaluation: the report of %s cannot be read", record)
	}
	dataset, found, err := artifact.ReadContent(ctx, store, evaluated.Dataset)
	if err != nil || !found {
		return Reasked{}, errors.Join(err, fmt.Errorf("evaluation: the dataset of %s cannot be read", record))
	}
	var cases []MultipleChoiceCase
	if err := json.Unmarshal(dataset.Data, &cases); err != nil {
		return Reasked{}, fmt.Errorf("evaluation: the dataset of %s holds no choices: %w", record, err)
	}
	byName := make(map[string]MultipleChoiceCase, len(cases))
	for _, testCase := range cases {
		byName[testCase.Name] = testCase
	}
	var answers struct {
		Observations []struct {
			Name     string `json:"name"`
			Selected int    `json:"selected"`
			Raw      string `json:"raw"`
		} `json:"observations"`
	}
	if err := json.Unmarshal(report.Data, &answers); err != nil {
		return Reasked{}, err
	}
	var reasked Reasked
	for _, answer := range answers.Observations {
		var cause UnreadCauses
		if answer.Selected != unmatchedChoice {
			continue
		}
		if cause.add(answer.Raw); cause.Budget == 0 {
			continue
		}
		testCase, held := byName[answer.Name]
		if !held {
			return reasked, fmt.Errorf("evaluation: %s's dataset holds no case %s", record, answer.Name)
		}
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
