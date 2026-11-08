package evaluation

import (
	"cmp"
	"context"

	"overgo/internal/artifact"
)

// keptAnswer is one case's answer as a generated-answer evaluation keeps it: what was written, its tokens, and for a
// choice the probability of the right one.
type keptAnswer struct {
	Text            string   `json:"text"`
	PromptTokens    int      `json:"prompt_tokens"`
	GeneratedTokens int      `json:"generated_tokens"`
	Probability     *float64 `json:"probability,omitempty"`
}

// judgedAsked names what a judged case's answer depends on: the judge, the code that answers and where it ran, and the
// case as the judge is asked it, whose prompt names its picture or clip by a reference that holds the media's digest
// and the judge's own identity (MediaInput.Reference). The model that drew the media is no part of it, so pictures a
// reference drew, judged again for each candidate compared with it, are not asked again; the code is the identity the
// campaign's build counted where it counted one (AnsweredBy), else its commit.
func (p Plan) judgedAsked(judge string, testCase GeneratedAnswerCase) (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, struct {
		Judge       string      `json:"judge"`
		Code        string      `json:"code"`
		Environment artifact.ID `json:"environment"`
		Prompt      string      `json:"prompt"`
		MaxTokens   int         `json:"max_tokens"`
		Choices     []string    `json:"choices,omitempty"`
	}{judge, cmp.Or(p.answered, p.body.CodeCommit), p.body.Environment, testCase.Prompt, testCase.MaxTokens, testCase.Choices})
}

// replayJudged is the answer the judge gave the case before, kept under what it depends on (judgedAsked), or else the
// one ask gives, which is kept so: a judged question costs 6 to 27 s of the card, and the 415 questions of TIFA's
// pictures about three hours. It says whether the answer was kept from before.
func replayJudged(ctx context.Context, repository artifact.Repository, plan Plan, judge string, testCase GeneratedAnswerCase, ask func() (keptAnswer, error)) (keptAnswer, bool, error) {
	observed, err := plan.judgedAsked(judge, testCase)
	if err != nil {
		return keptAnswer{}, false, err
	}
	held, err := loadObserved[keptAnswer](ctx, repository, observed)
	if err != nil {
		return keptAnswer{}, false, err
	}
	before := len(held.values) > 0
	answer, err := held.answer(ctx, 0, ask)
	if err != nil {
		return keptAnswer{}, false, err
	}
	return answer, before, held.write(ctx)
}
