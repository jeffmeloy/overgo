package evaluation

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// writtenAnswer is one answer a counting fixture writes: its text, and how many tokens it is written in.
type writtenAnswer struct {
	text   string
	tokens int
}

// countedChat answers a chat-template choice with each of its answers in turn, written in the tokens the answer
// names, as a model whose turn ends at a token count of its own does.
type countedChat struct {
	answers []writtenAnswer
	calls   int
}

func (c *countedChat) ShapeChatPrompt(prompt string) (string, error) {
	return "<user>" + prompt + "<model>", nil
}

func (c *countedChat) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	answer := c.answers[c.calls%len(c.answers)]
	c.calls++
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 1})
	}
	if options.OnToken != nil && answer.text != "" {
		if err := options.OnToken(inference.TokenEvent{Piece: answer.text}); err != nil {
			return nil, "", err
		}
	}
	return make([]tokenizer.TokenID, 1+answer.tokens), answer.text, nil
}

func (c *countedChat) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	panic("chat-template scoring must not score likelihoods")
}

// TestABudgetEndIsCalledOnlyWhereTheTokensRanOut reads a chat-template choice reading through a campaign and the
// evidence index, as a measurement run and the scorecard do (SQA-362). Of four answers, one names its letter; one is
// MiniCPM5-1B's " 11.9, which corresponds to", written in the protocol's eight tokens; one is the same text from a
// model that ended its turn after five; one is no text at all, a turn ended at once. All three that name no option
// are counted, and only the one whose record shows the eight tokens written is called budget-ended: the other two are
// of a cause the record does not show, whatever their text resembles.
func TestABudgetEndIsCalledOnlyWhereTheTokensRanOut(t *testing.T) {
	t.Parallel()
	suite := chatLetterSuite()
	suite.Cases = append(suite.Cases, MultipleChoiceCase{Name: "four", Prompt: "Q4\nAnswer:", Candidates: []string{" A", " B"}, Answer: 1})
	runtime := &countedChat{answers: []writtenAnswer{
		{"B) two", 2}, {" 11.9, which corresponds to", 8}, {" 11.9, which corresponds to", 5}, {"", 1},
	}}
	reading := indexedChatReading(t, runtime, suite)
	if reading.Unread != 3 {
		t.Fatalf("the reading counts %d answers that named no option, want the three", reading.Unread)
	}
	if causes := reading.UnreadCauses; causes.Budget != 1 || causes.Thinking != 0 || causes.Declined != 0 || causes.Repeated != 0 ||
		!strings.Contains(causes.String(), "2 of a cause the record does not show") {
		t.Errorf("the answers that named no option sort as %+v (%s), want one budget-ended by its tokens and two of a cause the record does not show", causes, causes)
	}
}
