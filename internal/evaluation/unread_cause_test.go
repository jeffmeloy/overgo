package evaluation

import (
	"encoding/json"
	"testing"

	"overgo/internal/testcheck"
)

// TestAnAnswerNamingNoOptionIsSortedByItsCause sorts the MMLU-Pro answers that named no option by why
// (answer-format/unanswered), on texts the models wrote under the chat-template protocol on 2026-11-07: Qwen3.5-9B and
// both gemma-4-12B-it builds began working the problem ("calculated as follows") and MiniCPM5-1B gave the value ("11.9,
// which corresponds to"), each cut by the protocol's eight tokens, overgo's request; gemma also said no option fits,
// the model's own. A thought left open and a phrase repeated three times are their own causes, and an answer that named
// a letter counts toward none.
func TestAnAnswerNamingNoOptionIsSortedByItsCause(t *testing.T) {
	t.Parallel()
	type observation struct {
		Selected int    `json:"selected"`
		Raw      string `json:"raw"`
	}
	report := testcheck.Must(json.Marshal(map[string][]observation{"observations": {
		{Selected: unmatchedChoice, Raw: " calculated as follows:\n\n1. "},
		{Selected: unmatchedChoice, Raw: " 11.9, which corresponds to"},
		{Selected: unmatchedChoice, Raw: " derived from the compound interest formula:\n"},
		{Selected: unmatchedChoice, Raw: " not explicitly listed among the options provided because"},
		{Selected: unmatchedChoice, Raw: "<think>\nLet me work through the interest"},
		{Selected: unmatchedChoice, Raw: "To find the find the find the degree of the degree of the degree of the"},
		{Selected: 2, Raw: " C"},
	}}))(t)
	want := UnreadCauses{Budget: 3, Thinking: 1, Declined: 1, Repeated: 1}
	if got := unreadCauses(report); got != want {
		t.Fatalf("the answers that named no option sort as %+v, want %+v", got, want)
	}
	if got := unreadCauses([]byte("not a report")); got != (UnreadCauses{}) {
		t.Fatalf("an unreadable report sorts as %+v, want nothing", got)
	}
}
