package main

import (
	"testing"

	"overgo/internal/evaluation"
)

// TestAnAnswerNamingNoOptionIsSortedByItsCause holds the evaluate command's reading to the sort
// (answer-format/unanswered): after a suite's metrics it says how many of its answers named no option and why, the
// request's causes first, as Qwen3.5-9B's MMLU-Pro reading would have said its 145; and it says nothing of a reading
// whose answers all named one.
func TestAnAnswerNamingNoOptionIsSortedByItsCause(t *testing.T) {
	t.Parallel()
	if line := unreadLine(evaluation.CampaignResult{}); line != "" {
		t.Errorf("a reading whose answers all named an option says %q", line)
	}
	line := unreadLine(evaluation.CampaignResult{Unread: evaluation.UnreadCauses{Budget: 69, Declined: 87}})
	if want := "  answers that named no option: 69 ran out of the answer's tokens, 87 said no option fits\n"; line != want {
		t.Errorf("the reading says %q, want %q", line, want)
	}
}
